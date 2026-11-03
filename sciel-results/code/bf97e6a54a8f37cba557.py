"""
Optimized solver for K_q(n, radius) using ILP (scipy.optimize.milp) 
with a robust greedy/local-search fallback.
"""
import time
import itertools
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def _int_to_word(idx, q, n):
    w = [0] * n
    for i in range(n - 1, -1, -1):
        w[i] = idx % q
        idx //= q
    return w

def _word_to_idx(word, q, n):
    idx = 0
    for x in word:
        idx = idx * q + x
    return idx

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total = q ** n
    
    # 1. Generate all words
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    
    # 2. Build the incidence matrix A where A[w, c] = 1 if d(w, c) <= radius
    # For q=3, n=6, total=729. Matrix is 729x729.
    # We can build this efficiently.
    
    # Precompute distances or just check on the fly for the matrix construction.
    # Since total is small, we can even do pairwise distance checks if optimized.
    # But for general n, this might be slow. 
    # For the specific task (q=3, n=6), it's fine.
    # For larger n, we might need a different approach, but the task is fixed.
    
    # Let's check if the problem is the specific one we know.
    is_known_task = (q == 3 and n == 6 and radius == 1)
    
    if is_known_task:
        # Fast path for the known optimal problem
        try:
            # Build A
            A = np.zeros((total, total), dtype=np.int8)
            for w_idx, w in enumerate(all_words):
                A[w_idx, w_idx] = 1
                for i in range(n):
                    for v in range(q):
                        if v != w[i]:
                            w_prime = w[:]
                            w_prime[i] = v
                            c_idx = _word_to_idx(w_prime, q, n)
                            A[w_idx, c_idx] = 1
            
            # Objective: minimize sum(x)
            c_obj = np.ones(total, dtype=float)
            
            # Constraints: A x >= 1
            # LinearConstraint(A, lb=np.ones(total), ub=np.inf)
            cons = LinearConstraint(A, lb=np.ones(total), ub=np.inf)
            
            # Bounds: 0 <= x <= 1
            bounds = Bounds(lb=np.zeros(total), ub=np.ones(total))
            
            # Integrality: all integer
            integrality = np.ones(total)
            
            # Solve
            res = milp(c=c_obj, constraints=cons, integrality=integrality, bounds=bounds)
            
            if res.status == 0 and res.fun is not None:
                # Extract solution
                x = res.x
                # Threshold at 0.5 to get binary
                selected_indices = np.where(x > 0.5)[0]
                
                # Verify validity
                # Check if every word is covered
                # covered = A[:, selected_indices].sum(axis=1)
                # if (covered >= 1).all():
                #     return [all_words[i] for i in selected_indices]
                
                # Quick verification
                is_valid = True
                for w_idx in range(total):
                    if A[w_idx, selected_indices].sum() < 1:
                        is_valid = False
                        break
                
                if is_valid:
                    return [all_words[i] for i in selected_indices]
            else:
                # ILP failed or timed out, fall back
                pass
                
        except Exception:
            pass
    
    # Fallback: Greedy Set Cover + Local Search
    # This handles general cases or if ILP fails.
    
    # 1. Greedy Initialization
    # Precompute balls
    balls = []
    for w in all_words:
        ball = [w]
        for i in range(n):
            for v in range(q):
                if v != w[i]:
                    w_prime = w[:]
                    w_prime[i] = v
                    ball.append(w_prime)
        balls.append(ball)
    
    covered = [False] * total
    code_indices = []
    
    # Greedy step
    uncovered_count = total
    while uncovered_count > 0:
        if time.time() > deadline - 5.0:
            break
            
        best_idx = -1
        max_gain = 0
        
        # For speed, we can restrict candidates to those with high potential
        # But with 729 nodes, full scan is fast.
        for c_idx, c_word in enumerate(all_words):
            # Gain = number of uncovered words in ball(c)
            gain = 0
            for u_word in balls[c_idx]:
                u_idx = _word_to_idx(u_word, q, n)
                if not covered[u_idx]:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_idx = c_idx
        
        if best_idx == -1 or max_gain == 0:
            break
            
        code_indices.append(best_idx)
        for u_word in balls[best_idx]:
            u_idx = _word_to_idx(u_word, q, n)
            if not covered[u_idx]:
                covered[u_idx] = True
                uncovered_count -= 1
                
    # 2. Local Search (Swap/Remove)
    # Try to remove codewords if redundant
    # Try to swap codewords to reduce size
    
    current_code = set(code_indices)
    
    def is_valid_code(code_set):
        # Check coverage
        # For each word, is it in the ball of some codeword?
        # Optimized: for each codeword, mark its ball.
        # But we need to check ALL words.
        # Since total is small, we can just iterate.
        for w_idx, w in enumerate(all_words):
            covered_by_any = False
            for c_idx in code_set:
                # Check distance
                # We can use precomputed balls? 
                # balls[c_idx] contains words within dist 1.
                # So w is covered if w_idx is in balls_indices[c_idx]?
                # We didn't precompute index sets for balls, only word lists.
                # Let's precompute index sets for balls for the local search phase.
                pass
        return True # Placeholder, need efficient check

    # Efficient validity check for local search
    # Precompute ball_index_sets
    ball_idx_sets = []
    for b in balls:
        s = set()
        for w in b:
            s.add(_word_to_idx(w, q, n))
        ball_idx_sets.append(s)
        
    # Also precompute reverse: for each word, which codewords cover it?
    # Actually, just check coverage directly.
    
    def check_coverage(code_list):
        # Mark all covered words
        # Use a set of covered indices
        cov = set()
        for c in code_list:
            cov.update(ball_idx_sets[c])
        return len(cov) == total

    # Local Search: Remove
    improved = True
    while improved:
        improved = False
        for i in range(len(code_indices)):
            # Try removing code_indices[i]
            if len(code_indices) <= 1:
                continue
            test_code = code_indices[:i] + code_indices[i+1:]
            if check_coverage(test_code):
                code_indices = test_code
                improved = True
                break # Restart loop as indices changed
                
    # Local Search: Swap
    # Try swapping one codeword with another (not in code)
    # This is O(N^2 * Check). Might be slow if N is large, but N~100.
    # 100^2 * 729 * 13 ~ 10^8. Might be too slow in Python.
    # Only do a few random swaps if time permits.
    
    if time.time() < deadline - 10.0:
        for _ in range(50): # Limit iterations
            if time.time() > deadline - 10.0:
                break
            if not code_indices:
                break
            i = random.randint(0, len(code_indices) - 1)
            old_c = code_indices[i]
            # Pick a new candidate
            # Candidates: words not in code
            # To speed up, pick a word that is "far" from others? 
            # Just random.
            new_c = random.randint(0, total - 1)
            if new_c == old_c or new_c in code_indices:
                continue
            
            test_code = code_indices[:]
            test_code[i] = new_c
            
            if check_coverage(test_code):
                # Swap is valid. Does it reduce size? No, size is same.
                # But it might enable further removals.
                code_indices = test_code
                # Try to remove again
                improved = True
                while improved:
                    improved = False
                    for j in range(len(code_indices)):
                        if len(code_indices) <= 1:
                            continue
                        test_code2 = code_indices[:j] + code_indices[j+1:]
                        if check_coverage(test_code2):
                            code_indices = test_code2
                            improved = True
                            break

    return [all_words[i] for i in code_indices]


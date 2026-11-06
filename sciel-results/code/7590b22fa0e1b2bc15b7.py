import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    MAX_VAL = 10000
    N = 12
    LIMIT = 2 * MAX_VAL

    # 1. SPF Table for fast factorization
    spf = list(range(LIMIT + 1))
    for i in range(2, int(LIMIT**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def get_prime_factors(n):
        factors = set()
        while n > 1:
            factors.add(spf[n])
            n //= spf[n]
        return factors

    def count_primes(arr):
        union = set()
        for i in range(N):
            for j in range(i+1, N):
                union |= get_prime_factors(arr[i] + arr[j])
        return len(union)

    # Strategy:
    # 1. Try to find a set where all sums are P-smooth for a small prime set P.
    #    This minimizes the union of prime factors to exactly P.
    #    We try P = {2,3}, {2,3,5}, {2,3,5,7}, etc.
    #    Use the reduction: a_i = M * b_i, where M = prod(P).
    #    Then a_i + a_j = M(b_i + b_j).
    #    We need b_i + b_j to be P-smooth.
    #    This is a clique problem on the graph of b in [1, MAX_VAL//M].
    
    # Candidate prime sets in order of preference (smaller is better)
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
    ]

    best_result = None
    best_score = float('inf')

    for P in prime_sets:
        if time.time() > deadline - 5.0:
            break
            
        M = 1
        for p in P:
            M *= p
            
        max_b = MAX_VAL // M
        if max_b < N:
            continue

        # Precompute which sums of b's are P-smooth
        # s = b_i + b_j, max s = 2 * max_b
        is_smooth = [False] * (2 * max_b + 1)
        for s in range(2, 2 * max_b + 1):
            n = s
            valid = True
            while n > 1:
                p = spf[n]
                if p not in P:
                    valid = False
                    break
                while n % p == 0:
                    n //= p
            is_smooth[s] = valid

        # Build adjacency list for the graph of b's
        # neighbors[b] = set of c != b such that is_smooth[b+c]
        # To save memory and time, we can just check on the fly or use a bitset/list
        # Given max_b is at most 5000 (for M=2), 2*max_b is 10000.
        # Checking is_smooth[b+c] is O(1).
        
        # We will use a randomized greedy backtracking search to find a clique of size N.
        # Order nodes by degree (number of neighbors) to prune better?
        # Or just random restarts with a greedy heuristic.
        
        # Let's try a few restarts.
        found_clique = None
        random.seed(42)
        
        # Pre-compute neighbors for speed?
        # max_b can be up to 5000. 5000*5000 = 25M checks. Might be slow in Python if not optimized.
        # But is_smooth is a list lookup.
        # Let's compute degrees first to prioritize high-degree nodes.
        degrees = [0] * (max_b + 1)
        for b in range(1, max_b + 1):
            for c in range(1, max_b + 1):
                if b != c and is_smooth[b+c]:
                    degrees[b] += 1
        
        # Sort indices by degree descending
        sorted_indices = sorted(range(1, max_b + 1), key=lambda x: degrees[x], reverse=True)
        
        # Try to find clique using backtracking with forward checking
        # We'll pick nodes from sorted_indices.
        
        def find_clique_k(size):
            # Backtracking search
            # We want a set of 'size' distinct b's such that all pairwise sums are smooth.
            # We build the set one by one.
            # candidates: list of b's that can be added to current_set
            # To be fast, we restrict candidates to those with high degree.
            
            # We'll use a recursive function or iterative with stack.
            # Given small N (12), recursion depth is fine.
            
            # Initial candidates: all b's
            # But we can prune: if we pick b1, b2 must be neighbor of b1.
            
            # Let's use a simple randomized greedy with backtracking.
            # Pick a node, then intersect candidates with its neighbors.
            
            # To improve success rate, we can try multiple random orders or starts.
            
            # We'll implement a specific search:
            # 1. Pick a starting node from the top-degree nodes.
            # 2. Candidates = neighbors of start.
            # 3. Pick next node from candidates (prefer high degree within candidates).
            # 4. Candidates = intersection of neighbors of all chosen nodes.
            # 5. If stuck, backtrack.
            
            # For simplicity and speed, let's try a few random greedy runs first.
            for _ in range(200):
                if time.time() > deadline - 2.0:
                    return None
                
                # Pick a random start from top 10% degrees
                top_k = max(1, len(sorted_indices) // 10)
                start_idx = random.randint(0, top_k - 1)
                start = sorted_indices[start_idx]
                
                clique = [start]
                candidates = set()
                for c in range(1, max_b + 1):
                    if c != start and is_smooth[start + c]:
                        candidates.add(c)
                
                for _ in range(size - 1):
                    if not candidates:
                        break
                    
                    # Pick a node from candidates
                    # Prefer high degree in the original graph? Or degree within candidates?
                    # Degree within candidates is better for clique finding.
                    # Let's just pick randomly for now, or highest original degree.
                    # Random is often better for diversity in restarts.
                    next_node = random.choice(list(candidates))
                    clique.append(next_node)
                    
                    # Update candidates
                    new_candidates = set()
                    for c in candidates:
                        if c != next_node and is_smooth[next_node + c]:
                            new_candidates.add(c)
                    candidates = new_candidates
                    
                if len(clique) == size:
                    return clique
            return None

        # Try to find clique
        found_clique = find_clique_k(N)
        
        if found_clique:
            # Convert to a_i
            result = [M * b for b in found_clique]
            # Verify
            valid = True
            primes_used = set()
            for i in range(N):
                for j in range(i+1, N):
                    s = result[i] + result[j]
                    pf = get_prime_factors(s)
                    if not pf.issubset(set(P)):
                        valid = False
                        break
                    primes_used |= pf
                if not valid:
                    break
            
            if valid:
                score = len(primes_used)
                if score < best_score:
                    best_score = score
                    best_result = result
                # Since we iterate P in increasing size, the first valid one is optimal.
                return result

    # Fallback: If no pure P-smooth solution found, use local search
    # Initialize with a structured set (e.g., multiples of 210)
    if best_result is None:
        # Multiples of 210
        # 210 * 1, 210 * 2, ...
        # 210 * 47 = 9870
        # We need 12 distinct.
        init_set = [210 * i for i in range(1, 13)]
        # Verify they are <= 10000
        if max(init_set) > MAX_VAL:
            init_set = [210 * i for i in range(1, 13)] # 210*12 = 2520, safe.
        
        best_result = init_set
        best_score = count_primes(best_result)
        
        # Local Search
        current = best_result.copy()
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(N):
                old = current[i]
                # Try a few random replacements
                for _ in range(20):
                    new_val = random.randint(1, MAX_VAL)
                    if new_val in current:
                        continue
                    current[i] = new_val
                    new_score = count_primes(current)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    current[i] = old
                if improved:
                    break
        
        if current_score < best_score:
            best_result = current
            best_score = current_score

    return best_result

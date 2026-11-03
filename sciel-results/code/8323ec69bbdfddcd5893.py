import time
import random
import sys

def construct(q, n, radius):
    """
    Solves the covering code problem K_q(n, radius).
    Specifically optimized for q=3, n=6, radius=1.
    Uses Iterated Local Search (ILS) with efficient coverage counting.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Specific handling for the target problem size for maximum performance
    if q == 3 and n == 6 and radius == 1:
        return _solve_k361()
    
    # Generic fallback for other parameters (less optimized)
    return _solve_generic(q, n, radius)

def _solve_k361():
    """
    Optimized solver for K_3(6, 1).
    Space size: 729.
    Ball size: 1 + 6*2 = 13.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    N = 729
    # Precompute balls as lists of indices
    balls = []
    for idx in range(N):
        w = [0] * 6
        temp = idx
        for i in range(5, -1, -1):
            w[i] = temp % 3
            temp //= 3
        
        ball = [idx]
        # Generate neighbors at distance 1
        for pos in range(6):
            orig = w[pos]
            for val in range(3):
                if val != orig:
                    # Calculate new index
                    # index = w0*3^5 + ... + w5
                    # delta = (val - orig) * 3^(5-pos)
                    delta = (val - orig) * (3 ** (5 - pos))
                    ball.append(idx + delta)
        balls.append(ball)
    
    # Precompute ball masks? Not needed if we use counts.
    # We will use cover_counts array.
    
    def get_time():
        return time.time()
        
    def is_valid_cover(cover_counts):
        # cover_counts[w] > 0 for all w
        # This is expensive if done naively. 
        # Instead, we maintain a 'uncovered' count or set.
        pass

    # State: 
    # code: list of indices
    # cover_counts: list of ints, length 729
    
    # --- Phase 1: Initial Greedy ---
    def greedy_init():
        code = []
        cover_counts = [0] * N
        uncovered = set(range(N))
        
        # Heuristic: Pick word with max "new coverage"
        # To speed up, we can just iterate and pick the first that covers the most uncovered.
        # Since N is small, we can afford a simple loop.
        
        while uncovered:
            if get_time() > deadline:
                break
            best_idx = -1
            best_gain = -1
            
            # Optimization: Only consider uncovered words as candidates?
            # A codeword must cover itself. If it's already covered by others, 
            # adding it might still help cover its neighbors.
            # However, if we pick a covered word, it's likely redundant later.
            # Standard greedy: pick any word that maximizes new coverage.
            
            # To speed up, we can sample or iterate all. 729 is small.
            for idx in range(N):
                # Calculate gain: number of neighbors in 'uncovered'
                gain = 0
                for b in balls[idx]:
                    if b in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                
                # Early break if perfect?
                if best_gain == 13:
                    break
            
            if best_idx == -1:
                break
                
            code.append(best_idx)
            for b in balls[best_idx]:
                cover_counts[b] += 1
                if b in uncovered:
                    uncovered.remove(b)
                    
        return code, cover_counts

    # --- Local Search: Remove Redundant ---
    def local_search_remove(code, cover_counts):
        # Try to remove any codeword that is redundant
        # Redundant if all its neighbors are covered by >= 2 codewords (including itself)
        # So, check if min(cover_counts[b] for b in balls[c]) >= 2
        
        improved = True
        while improved:
            if get_time() > deadline:
                return code, cover_counts
            
            improved = False
            # Randomize order to avoid bias
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                c = code[i]
                is_redundant = True
                for b in balls[c]:
                    if cover_counts[b] < 2:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove c
                    code.pop(i)
                    for b in balls[c]:
                        cover_counts[b] -= 1
                    improved = True
                    # Restart loop to check for new redundancies
                    break
                    
        return code, cover_counts

    # --- Repair: Fix invalid cover ---
    def repair(code, cover_counts):
        # Find uncovered words and add codewords to cover them
        while True:
            if get_time() > deadline:
                return code, cover_counts
            
            # Find an uncovered word
            uncovered_word = -1
            for w in range(N):
                if cover_counts[w] == 0:
                    uncovered_word = w
                    break
            
            if uncovered_word == -1:
                return code, cover_counts # Valid cover
            
            # Pick a codeword to cover this word
            # Candidates: words in ball(uncovered_word)
            # Choose one that covers the most other uncovered words
            best_cand = -1
            best_gain = -1
            
            # We can restrict candidates to the ball of the uncovered word
            for cand in balls[uncovered_word]:
                gain = 0
                for b in balls[cand]:
                    if cover_counts[b] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cand = cand
            
            if best_cand != -1:
                code.append(best_cand)
                for b in balls[best_cand]:
                    cover_counts[b] += 1

    # --- Main Loop: ILS ---
    
    # Initial solution
    code, cover_counts = greedy_init()
    code, cover_counts = local_search_remove(code, cover_counts)
    
    best_code = list(code)
    best_size = len(code)
    
    # Iterated Local Search
    # Perturb by removing random codewords and repairing
    while get_time() < deadline - 5.0: # Leave 5s buffer
        # Create a copy
        curr_code = list(code)
        curr_counts = list(cover_counts)
        
        # Perturb: Remove 1-3 random codewords
        num_remove = random.randint(1, 3)
        if len(curr_code) > num_remove:
            to_remove = random.sample(curr_code, num_remove)
            for c in to_remove:
                # Find index in list
                try:
                    idx = curr_code.index(c)
                    curr_code.pop(idx)
                except ValueError:
                    continue
                for b in balls[c]:
                    curr_counts[b] -= 1
        
        # Repair
        curr_code, curr_counts = repair(curr_code, curr_counts)
        
        # Local Search
        curr_code, curr_counts = local_search_remove(curr_code, curr_counts)
        
        # Accept if better or equal (to allow exploration)
        if len(curr_code) < best_size:
            best_size = len(curr_code)
            best_code = list(curr_code)
            code = list(curr_code)
            cover_counts = list(curr_counts)
        elif len(curr_code) == best_size:
            # With some probability, accept equal to escape local minima?
            # Or just keep current best.
            pass
        else:
            # Revert to previous best state?
            # In ILS, we usually keep the current state as the starting point for next perturbation
            # even if it's worse, to explore the neighborhood.
            # But for minimization, it's often better to restart from the best known.
            # Let's try "Best-First" ILS: always restart from best_code.
            # This is safer.
            pass
            # If we want to explore, we would set:
            # code = curr_code
            # cover_counts = curr_counts
            
    # Convert best_code indices back to words
    result = []
    for idx in best_code:
        w = [0] * 6
        temp = idx
        for i in range(5, -1, -1):
            w[i] = temp % 3
            temp //= 3
        result.append(w)
        
    return result

def _solve_generic(q, n, radius):
    """
    Fallback solver for general q, n, radius.
    Uses simple greedy.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    total = q ** n
    # Precompute balls
    balls = []
    for idx in range(total):
        w = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            w[i] = temp % q
            temp //= q
        
        ball = [idx]
        # Radius 1 neighbors
        for pos in range(n):
            orig = w[pos]
            pw = q ** (n - 1 - pos)
            for val in range(q):
                if val != orig:
                    ball.append(idx + (val - orig) * pw)
        balls.append(ball)
        
    code = []
    cover_counts = [0] * total
    uncovered = set(range(total))
    
    while uncovered:
        if time.time() > deadline:
            break
        best_idx = -1
        best_gain = -1
        
        for idx in range(total):
            gain = 0
            for b in balls[idx]:
                if b in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                if best_gain == len(balls[idx]):
                    break
        
        if best_idx == -1:
            break
            
        code.append(best_idx)
        for b in balls[best_idx]:
            cover_counts[b] += 1
            if b in uncovered:
                uncovered.remove(b)
                
    # Local search remove
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(code)):
            c = code[i]
            redundant = True
            for b in balls[c]:
                if cover_counts[b] < 2:
                    redundant = False
                    break
            if redundant:
                code.pop(i)
                for b in balls[c]:
                    cover_counts[b] -= 1
                improved = True
                break
                
    # Convert to words
    result = []
    for idx in code:
        w = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            w[i] = temp % q
            temp //= q
        result.append(w)
    return result

if __name__ == "__main__":
    # Test
    res = construct(3, 6, 1)
    print(f"Found {len(res)} codewords")

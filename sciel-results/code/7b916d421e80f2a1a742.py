import itertools
import random
import time
import sys

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    N = q ** n
    # Precompute neighbors (ball of radius 1) for each word
    # Index mapping: tuple (a0, a1, ..., a5) -> a0*3^5 + a1*3^4 + ... + a5
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[n - 1 - i] = idx % q
            idx //= q
        return tuple(t)

    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    words = [tuple(t) for t in itertools.product(range(q), repeat=n)]
    
    # neighbors[i] is a list of indices within distance 1 from words[i]
    neighbors = [[] for _ in range(N)]
    # neighbors_set[i] is a set for fast lookup
    neighbors_set = [set() for _ in range(N)]
    
    for idx, w in enumerate(words):
        # Self
        neighbors[idx].append(idx)
        # Neighbors
        for i in range(n):
            for d in range(1, q):
                new_w = list(w)
                new_w[i] = (new_w[i] + d) % q
                neighbors[idx].append(tuple_to_idx(tuple(new_w)))
        # Deduplicate just in case (though logic above is distinct)
        neighbors[idx] = list(set(neighbors[idx]))
        neighbors_set[idx] = set(neighbors[idx])
    
    # Precompute for each word, the list of words that cover it (i.e., are in its ball)
    # coverers[idx] = list of codeword indices c such that idx is in neighbors[c]
    # This is the reverse of neighbors.
    # Since neighbors is symmetric for Hamming distance 1, neighbors[i] == coverers[i] in terms of set content?
    # No. neighbors[i] is the ball around i.
    # If c is a codeword, it covers u if u is in neighbors[c].
    # So for a fixed u, the codewords that cover u are those c such that u in neighbors[c].
    # This is exactly neighbors[u] because the relation "distance 1" is symmetric.
    # So coverers[u] = neighbors[u].
    
    def greedy_init():
        """
        Greedy set cover heuristic.
        """
        code = []
        code_set = set()
        covered = [False] * N
        
        # Random order for tie-breaking or initial selection
        # We pick the word that covers the most *new* uncovered words.
        
        # Optimization: Maintain a count of how many codewords cover each word?
        # Or just track uncovered set.
        
        # Simple approach:
        # 1. Count uncovered neighbors for each candidate.
        # 2. Pick best.
        # 3. Mark covered.
        
        # To speed this up, we can maintain an array `uncovered_count`?
        # Actually, standard greedy set cover is O(N * |Code| * BallSize).
        # N=729, BallSize=19.
        
        while True:
            # Check if all covered
            # This check is O(N).
            # We can maintain a counter `num_uncovered`.
            
            best_w = -1
            best_count = 0
            
            # Iterate through all words to find the best candidate
            # We can skip words that are already in code?
            # Yes.
            
            # Optimization: Only consider words that are not in code_set.
            # Even better: Only consider words that have at least one uncovered neighbor.
            
            # Let's just loop through all N words.
            for w in range(N):
                if w in code_set:
                    continue
                cnt = 0
                for u in neighbors[w]:
                    if not covered[u]:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_w = w
            
            if best_w == -1 or best_count == 0:
                break
            
            code.append(best_w)
            code_set.add(best_w)
            
            for u in neighbors[best_w]:
                covered[u] = True
            
            # Early exit if all covered
            # We can check `all(covered)` but that's O(N).
            # Since we just marked some, we can just check if any uncovered remains?
            # It's probably fine to just let the loop run until best_count is 0.
            
        return code, code_set, covered

    def local_search(code, code_set, time_limit):
        """
        Local search to reduce code size.
        Strategies:
        1. Remove: If a codeword covers no unique words, remove it.
        2. Swap: If a codeword covers unique words, see if another word can cover them
           while covering at least as many other words (or just covering the unique ones).
           Actually, a swap is valid if we can replace c with c' such that:
           - All words uniquely covered by c are covered by c'.
           - (Optional) c' might cover other things, but we don't lose coverage.
           - We want to minimize size, so swap doesn't change size.
           - But swap might enable future removals.
           
           More aggressive:
           3. Add+Remove: Add a word, then remove multiple words. (Complex)
           
           Let's stick to Remove and Swap.
        """
        # cover_count[u] = number of codewords in `code` that cover u
        cover_count = [0] * N
        for c in code:
            for nb in neighbors[c]:
                cover_count[nb] += 1
        
        best_code = list(code)
        best_size = len(best_code)
        start = time.time()
        
        while time.time() < start + time_limit:
            improved = False
            
            # Phase 1: Removal
            # Try to remove any codeword that is not essential.
            # Essential if there exists u in neighbors[c] with cover_count[u] == 1.
            
            # We iterate through the code.
            # If we remove one, we break and restart the loop (since cover_count changes).
            
            removed = False
            for i in range(len(code)):
                c = code[i]
                # Check if c is essential
                essential = False
                for nb in neighbors[c]:
                    if cover_count[nb] == 1:
                        essential = True
                        break
                
                if not essential:
                    # Remove c
                    code.pop(i)
                    code_set.remove(c)
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    improved = True
                    removed = True
                    break
            
            if removed:
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                continue
            
            # Phase 2: Swap
            # If we couldn't remove, try to swap.
            # For each c, find the set of unique words U_c = {u in neighbors[c] | cover_count[u] == 1}.
            # If U_c is empty, we would have removed c in Phase 1. So U_c is non-empty.
            # We want to find c' not in code such that U_c is a subset of neighbors[c'].
            # If such c' exists, we can swap c -> c'.
            # Does this help? It keeps size same. But it might allow a removal later.
            
            swapped = False
            # Randomize order to escape local minima?
            # The code list order is fixed. Let's shuffle a copy of indices to try?
            # Or just iterate.
            
            for i in range(len(code)):
                if time.time() > start + time_limit:
                    break
                c = code[i]
                
                # Find unique words covered by c
                unique_words = []
                for nb in neighbors[c]:
                    if cover_count[nb] == 1:
                        unique_words.append(nb)
                
                # unique_words is guaranteed non-empty (else Phase 1 would have caught it)
                
                # Find candidates c' that cover all unique_words
                # c' must be in neighbors[u] for all u in unique_words.
                # So c' is in the intersection of neighbors[u] for u in unique_words.
                # Also c' must not be in code_set (and c' != c, though c is in code_set).
                
                # Start with the first unique word's neighbors
                candidates = set(neighbors[unique_words[0]])
                for u in unique_words[1:]:
                    candidates &= neighbors_set[u]
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Remove existing codewords from candidates
                candidates -= code_set
                
                if not candidates:
                    continue
                
                # Pick a random candidate
                c_prime = random.choice(list(candidates))
                
                # Perform swap
                # Update cover_count
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                for nb in neighbors[c_prime]:
                    cover_count[nb] += 1
                
                code[i] = c_prime
                code_set.remove(c)
                code_set.add(c_prime)
                
                improved = True
                swapped = True
                break
            
            if not improved:
                # No removals, no swaps. Stuck in local minimum.
                # We can try a random restart or perturbation.
                # For now, break.
                break
        
        return best_code

    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    global_best = None
    global_best_size = float('inf')
    
    # We will run multiple iterations of Greedy + Local Search
    # Since the problem is small (N=729), greedy is fast.
    # Local search can be run for a while.
    
    # Strategy:
    # 1. Run greedy init.
    # 2. Run local search with a time budget.
    # 3. Update best.
    # 4. Repeat if time remains.
    
    # To diversify, we can randomize the greedy init slightly.
    # E.g., random tie-breaking in greedy.
    
    def randomized_greedy_init():
        code = []
        code_set = set()
        covered = [False] * N
        
        while True:
            best_w = -1
            best_count = 0
            best_candidates = []
            
            for w in range(N):
                if w in code_set:
                    continue
                cnt = 0
                for u in neighbors[w]:
                    if not covered[u]:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_candidates = [w]
                elif cnt == best_count and cnt > 0:
                    best_candidates.append(w)
            
            if best_w == -1 or best_count == 0:
                break
            
            best_w = random.choice(best_candidates)
            code.append(best_w)
            code_set.add(best_w)
            
            for u in neighbors[best_w]:
                covered[u] = True
            
        return code, code_set

    # Run several rounds
    # Each round: Init + Local Search
    # Budget per round: say 10s?
    # Total time 145s.
    
    # Let's do dynamic budgeting.
    # First few rounds short, later ones longer?
    # Or just fixed 10s per round.
    
    while time.time() < deadline - 5.0:
        code, code_set = randomized_greedy_init()
        
        # Allocate remaining time minus a safety buffer to local search
        remaining = deadline - time.time() - 2.0
        if remaining < 1.0:
            break
            
        # Cap local search time per iteration to allow multiple iterations
        # If we have lots of time, we can do longer searches.
        # But shorter searches allow more diversity.
        # Let's try 5s per local search, allowing ~25 iterations.
        search_time = min(5.0, remaining)
        
        best_code = local_search(code, code_set, search_time)
        
        if len(best_code) < global_best_size:
            global_best_size = len(best_code)
            global_best = best_code
        
        # If we found 73, we can stop early?
        # 73 is best known. 71 is lower bound.
        # If we find 73, it's likely optimal or near-optimal.
        # We can continue to try for 72 or 71, but it's hard.
        # Let's keep searching to be safe, but maybe reduce search time if we hit 73?
        # No, just keep going.
        
    if global_best is None:
        code, _ = randomized_greedy_init()
        global_best = code
        
    return [list(idx_to_tuple(c)) for c in global_best]


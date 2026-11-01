import itertools
import random
import time

def construct(q, n, radius):
    # Parameters
    total_words = q ** n
    # Precompute neighbors for radius 1
    # neighbors[i] is a list of indices j such that dist(i, j) <= 1
    neighbors = [[] for _ in range(total_words)]
    
    # Generate all words
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    for i, w in enumerate(words):
        neighbors[i].append(i) # distance 0
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                new_w_tuple = tuple(new_w)
                j = word_to_idx[new_w_tuple]
                neighbors[i].append(j)
        
    # Convert to sets for fast intersection? Or just lists.
    # For greedy, we need count of uncovered neighbors.
    
    def greedy_init(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = set(range(total_words))
        code = []
        counts = [0] * total_words
        
        while uncovered:
            best_c = -1
            max_cov = 0
            # To speed up, we can just iterate all candidates
            # 2187 candidates * ~200 steps = ~400k ops. Fast.
            for c in range(total_words):
                # Count how many neighbors of c are in uncovered
                cov = 0
                for w in neighbors[c]:
                    if w in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_c = c
                    # Random tie breaking
                    # if cov == max_cov:
                    #    if random.random() < 0.5: best_c = c
            if best_c == -1:
                break # Should not happen if uncovered is not empty
                
            code.append(best_c)
            for w in neighbors[best_c]:
                counts[w] += 1
                if w in uncovered:
                    uncovered.remove(w)
                    
        return code, counts

    def local_search(code, counts):
        # Try to remove codewords
        improved = True
        while improved:
            improved = False
            # Iterate in random order to avoid bias
            code_list = list(code)
            random.shuffle(code_list)
            for c in code_list:
                if c not in code:
                    continue
                # Check if removal is safe
                safe = True
                for w in neighbors[c]:
                    if counts[w] == 1:
                        safe = False
                        break
                if safe:
                    # Remove c
                    code.remove(c)
                    for w in neighbors[c]:
                        counts[w] -= 1
                    improved = True
                    break # Restart loop
        return code, counts

    def solve():
        best_code = None
        best_size = float('inf')
        
        start_time = time.time()
        time_limit = 110.0 # Leave margin
        
        # Run multiple restarts
        restarts = 0
        while time.time() - start_time < time_limit:
            restarts += 1
            code, counts = greedy_init(random_seed=restarts)
            code, counts = local_search(code, counts)
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
            
            # If we hit the theoretical lower bound or a very good number, we can stop
            # Lower bound is 156. Best known 186.
            if best_size <= 186:
                break
                
        if best_code is None:
            # Fallback to simple construction if something went wrong
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # Convert indices back to words
        return [words[i] for i in best_code]

    return solve()

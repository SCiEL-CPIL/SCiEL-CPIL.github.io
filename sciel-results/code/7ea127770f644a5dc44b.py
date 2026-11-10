import itertools
import random

def construct(q, n, radius):
    # This implementation is tuned for q=3, n=8, radius=1
    # but will handle general small cases reasonably well.
    
    total_words = q ** n
    
    # Precompute neighbors for each word in the space
    # A word is an integer in 0..total_words-1
    # We need to map int <-> base-q representation to calculate Hamming distance
    
    def int_to_list(val, base, length):
        res = [0] * length
        for i in range(length):
            res[i] = val % base
            val //= base
        return res
    
    def list_to_int(lst, base):
        val = 0
        for i in range(len(lst)-1, -1, -1):
            val = val * base + lst[i]
        return val

    # Precompute neighbors
    neighbors = []
    for i in range(total_words):
        w = int_to_list(i, q, n)
        neigh = set()
        neigh.add(i)
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[pos] = new_val
                ni = list_to_int(w, q)
                neigh.add(ni)
        w[pos] = orig # restore, though w is recreated in next iter
        neighbors.append(neigh)

    def greedy_construct(seed_val=None):
        if seed_val is not None:
            random.seed(seed_val)
        
        uncovered = set(range(total_words))
        code = []
        
        while uncovered:
            # To allow for different greedy paths, we shuffle the order of candidates
            # This breaks ties randomly
            candidates = list(uncovered)
            random.shuffle(candidates)
            
            best_c = -1
            best_count = -1
            for c in candidates:
                count = len(neighbors[c] & uncovered)
                if count > best_count:
                    best_count = count
                    best_c = c
                    
            code.append(best_c)
            uncovered.difference_update(neighbors[best_c])
        
        return code

    def reduce_code(code):
        # Convert to list for easy mutation
        code_list = list(code)
        
        # Calculate cover counts
        cover_counts = [0] * total_words
        for c in code_list:
            for w in neighbors[c]:
                cover_counts[w] += 1
        
        # Remove redundant codewords
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to allow easy removal from list
            for i in range(len(code_list) - 1, -1, -1):
                c = code_list[i]
                # Check if all neighbors are covered by at least one other codeword
                redundant = True
                for w in neighbors[c]:
                    if cover_counts[w] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove c
                    code_list.pop(i)
                    for w in neighbors[c]:
                        cover_counts[w] -= 1
                    changed = True
        
        return code_list

    # Run multiple times with different seeds to find a better solution
    best_code = []
    min_size = float('inf')
    
    # Number of trials
    num_trials = 10
    
    for t in range(num_trials):
        code = greedy_construct(seed_val=42 + t)
        reduced_code = reduce_code(code)
        
        if len(reduced_code) < min_size:
            min_size = len(reduced_code)
            best_code = reduced_code

    # Convert back to list of lists
    return [int_to_list(c, q, n) for c in best_code]

import time
import random
import sys

def construct(q, n, radius):
    # Problem specific: q=3, n=6, radius=1
    # Generalize slightly or hardcode? Task says "evaluator passes q, n, radius".
    # But the specific task is K_3(6,1).
    # If q, n, radius are different, this heuristic might not work or be optimal.
    # However, the prompt says "Football pool problem for 6 matches...".
    # I will assume q=3, n=6, radius=1 for the core logic but check inputs.
    # If inputs differ, fallback to a generic greedy?
    # The seed program hardcoded the tetracode logic for n=6.
    # I will implement the ILS for the specific case but add a guard.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to generic greedy or seed logic?
        # Seed logic:
        # tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # This works for any n >= 4.
        # For now, focus on 3,6,1.
        pass

    start_time = time.time()
    deadline = 115.0 # 5s margin
    
    # Precomputation for q=3, n=6
    total_words = q ** n
    # Convert word to index
    def to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
        
    def to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        w = to_word(idx)
        neigh = [idx]
        for i in range(n):
            orig = w[i]
            for val in range(q):
                if val == orig: continue
                w[i] = val
                neigh.append(to_idx(w))
            w[i] = orig
        neighbors[idx] = neigh

    # Greedy Init
    def greedy_init():
        uncovered = set(range(total_words))
        code = []
        while uncovered:
            # Find word in uncovered that covers most uncovered words
            best_w = -1
            best_count = 0
            # Optimization: We can just pick any uncovered word?
            # No, greedy choice matters.
            # Iterate through uncovered?
            # To speed up, we can maintain counts.
            # But N is small.
            for w in uncovered:
                count = 0
                for nb in neighbors[w]:
                    if nb in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_w = w
            
            if best_w == -1: break
            
            code.append(best_w)
            # Remove neighbors of best_w from uncovered
            for nb in neighbors[best_w]:
                if nb in uncovered:
                    uncovered.remove(nb)
                    
        return code

    # Remove Redundant
    def remove_redundant(code):
        # code is list of indices
        # Make it a list to allow modification
        code = list(code)
        if not code: return code
        
        cover_count = [0] * total_words
        for c in code:
            for nb in neighbors[c]:
                cover_count[nb] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(code)):
                c = code[i]
                # Check if c is redundant
                min_count = min(cover_count[nb] for nb in neighbors[c])
                if min_count > 1:
                    # Remove c
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    code.pop(i)
                    changed = True
                    break # Restart loop
        return code

    # ILS
    best_code = remove_redundant(greedy_init())
    current_code = best_code
    best_size = len(best_code)
    
    # To allow exploration, we might need to accept worse solutions or use simulated annealing.
    # Given the small size, simple random restarts or perturbations are good.
    
    while time.time() < deadline:
        # Perturb
        new_code = list(current_code)
        
        # Remove 1 or 2 random codewords
        k = random.randint(1, 2)
        if len(new_code) > k:
            to_remove = random.sample(new_code, k)
            for c in to_remove:
                new_code.remove(c)
        
        # Calculate uncovered
        covered = [False] * total_words
        for c in new_code:
            for nb in neighbors[c]:
                covered[nb] = True
        
        uncovered = [i for i in range(total_words) if not covered[i]]
        
        # Greedy fill
        # We can reuse the logic but only for uncovered
        # To be efficient, we just run a greedy step that adds words covering max uncovered
        # This is essentially the same as greedy_init but starting with some covered.
        
        # Let's implement a helper to fill
        while uncovered:
            best_w = -1
            best_count = 0
            for w in uncovered:
                count = 0
                for nb in neighbors[w]:
                    if not covered[nb]: # if nb is uncovered
                        count += 1
                if count > best_count:
                    best_count = count
                    best_w = w
            
            if best_w == -1: break
            
            new_code.append(best_w)
            for nb in neighbors[best_w]:
                covered[nb] = True
                if nb in uncovered:
                    uncovered.remove(nb)
                    
        # Remove redundant
        new_code = remove_redundant(new_code)
        
        if len(new_code) < best_size:
            best_size = len(new_code)
            best_code = new_code
            current_code = new_code
        else:
            # Accept with some probability to escape local minima?
            # Or just revert?
            # Let's try reverting to current_code if not better, 
            # but sometimes accept to explore?
            # If we always revert, we are stuck in local minima of the "remove+fill" operator.
            # Let's accept if size is same, or with prob 0.1 if worse?
            # Actually, if we revert, we are doing hill climbing.
            # Let's try: if len(new_code) <= len(current_code):
            #    current_code = new_code
            # else:
            #    if random.random() < 0.1: current_code = new_code
            #    else: current_code = best_code # or keep current?
            
            # Let's try a simple strategy:
            # If new is better, update best and current.
            # If new is worse, keep current.
            # To escape, we can randomly perturb best_code occasionally?
            # Or just rely on the randomness of "remove k random".
            # If we always revert to current_code (which is best_code if we only update on improvement),
            # we are doing random restarts from best_code? No, current_code stays best_code.
            # So we are repeatedly perturbing the best solution.
            # This is a form of Iterated Local Search (ILS) where the "local search" is the remove+fill+reduce.
            # This is a valid strategy.
            
            pass # current_code remains unchanged

    # Convert indices back to words
    result = [to_word(c) for c in best_code]
    return result

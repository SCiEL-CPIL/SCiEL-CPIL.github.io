import itertools
import time
import random

def construct(q, n, radius):
    # Only handle the specific case efficiently, fallback for others
    if q != 3 or n != 6 or radius != 1:
        # Fallback to the original tetracode construction or a simple greedy
        # The task specifies q=3, n=6, radius=1 usually, but we should be robust.
        # For now, stick to the tetracode for non-matching params to be safe, 
        # or a basic greedy.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Fallback for small n
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=6, radius=1
    deadline = time.time() + 110.0  # Safety margin
    
    # 1. Map words to indices
    # Word (w0, ..., w5) -> idx = w0 * 3^5 + w1 * 3^4 + ...
    # Precompute balls
    # ball[i] is a set of indices covered by codeword i
    
    # We can represent the set of covered words as a Python `set` of integers.
    # 729 sets of size 13.
    
    balls = []
    for i in range(729):
        # Decode i to word
        w = [0] * 6
        temp = i
        for k in range(5, -1, -1):
            w[k] = temp % 3
            temp //= 3
        
        # Generate all words in distance 1
        covered = {i}
        for pos in range(6):
            orig_val = w[pos]
            for delta in [1, 2]:
                new_val = (orig_val + delta) % 3
                new_w = w[:]
                new_w[pos] = new_val
                # Encode new_w
                new_idx = 0
                for v in new_w:
                    new_idx = new_idx * 3 + v
                covered.add(new_idx)
        balls.append(covered)
    
    # 2. Greedy Set Cover
    # Start with all words uncovered
    uncovered = set(range(729))
    selected = []
    
    # To improve greedy, we can break ties randomly or by index.
    # Randomizing tie-breaking and running multiple times can help.
    
    best_solution = None
    best_size = 81 # Initial upper bound from tetracode
    
    # Run multiple greedy passes with random tie-breaking
    # Since 729 is small, we can run many passes.
    
    for run in range(50):
        if time.time() > deadline:
            break
            
        uncovered = set(range(729))
        current_selected = []
        
        while uncovered:
            # Find candidate that covers most uncovered
            max_covered = 0
            best_candidate = -1
            candidates = []
            
            for i in range(729):
                count = len(balls[i] & uncovered)
                if count > max_covered:
                    max_covered = count
                    candidates = [i]
                elif count == max_covered and count > 0:
                    candidates.append(i)
            
            if not candidates:
                break # Should not happen if uncovered is not empty
                
            # Pick randomly from best candidates to diversify
            c = random.choice(candidates)
            current_selected.append(c)
            uncovered -= balls[c]
            
        if len(current_selected) < best_size:
            best_size = len(current_selected)
            best_solution = current_selected.copy()
            
    # 3. Local Search / Pruning on the best solution found
    if best_solution:
        # Convert to set for faster lookup
        sol_set = set(best_solution)
        
        # Try to remove redundant codewords
        improved = True
        while improved:
            improved = False
            for c in list(sol_set):
                # Check if removing c still covers everything
                # This is expensive if done naively.
                # Optimization: Precompute coverage counts?
                # For now, check validity of sol_set \ {c}
                
                # We can check if there exists a word w that is ONLY covered by c.
                # If yes, we cannot remove c.
                # If no, we can remove c.
                
                # Efficient check:
                # For each word w in ball[c], check if there is another c' in sol_set \ {c} such that w in ball[c'].
                # If for all w in ball[c], such a c' exists, then c is redundant.
                
                redundant = True
                for w in balls[c]:
                    covered_by_other = False
                    for c_prime in sol_set:
                        if c_prime == c:
                            continue
                        if w in balls[c_prime]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        redundant = False
                        break
                
                if redundant:
                    sol_set.remove(c)
                    improved = True
                    best_size = len(sol_set)
        
        # Try to swap codewords to allow further removals?
        # This is more complex. Given the time, the pruning step often gets to the minimum.
        # The greedy usually finds a set of size ~75-78. Pruning might get it to 73-75.
        
        best_solution = list(sol_set)

    # If we didn't find anything better than 81, return the tetracode solution
    if not best_solution or len(best_solution) >= 81:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    # Convert indices back to words
    result = []
    for idx in best_solution:
        w = [0] * 6
        temp = idx
        for k in range(5, -1, -1):
            w[k] = temp % 3
            temp //= 3
        result.append(w)
        
    return result

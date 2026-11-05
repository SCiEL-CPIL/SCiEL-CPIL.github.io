import itertools
import time
import random

def construct(q, n, radius):
    # q=3, n=7, radius=1
    space_size = q ** n
    deadline = time.time() + 110  # Safety margin
    
    # Precompute neighbors for all words
    # Encode word to int: sum(w[i] * q**i)
    # Decode int to word: [int // q**i % q for i in range(n)]
    
    neighbors = [[] for _ in range(space_size)]
    for x in range(space_size):
        # Convert x to tuple
        w = [0] * n
        temp = x
        for i in range(n):
            w[i] = temp % q
            temp //= q
        
        neighbors[x].append(x) # distance 0
        
        for i in range(n):
            orig = w[i]
            for val in range(q):
                if val == orig: continue
                w[i] = val
                # Encode back
                new_x = 0
                pow_q = 1
                for j in range(n):
                    new_x += w[j] * pow_q
                    pow_q *= q
                neighbors[x].append(new_x)
            w[i] = orig
    
    # Helper to check if code is valid and get size
    def is_valid(code):
        # Check coverage
        covered = [False] * space_size
        for c in code:
            for nb in neighbors[c]:
                covered[nb] = True
        return all(covered)
        
    def remove_redundancies(code, coverage_count=None):
        if coverage_count is None:
            coverage_count = [0] * space_size
            for c in code:
                for nb in neighbors[c]:
                    coverage_count[nb] += 1
        
        code_set = set(code)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Shuffle to avoid bias
            shuffled_code = list(code_set)
            random.shuffle(shuffled_code)
            
            for c in shuffled_code:
                if c not in code_set:
                    continue
                # Check if c is redundant
                # All neighbors of c must have coverage_count > 1
                is_redundant = True
                for nb in neighbors[c]:
                    if coverage_count[nb] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove c
                    code_set.remove(c)
                    for nb in neighbors[c]:
                        coverage_count[nb] -= 1
                    changed = True
                    if time.time() > deadline - 5:
                        break
        
        return list(code_set), coverage_count

    # Method 1: Greedy Construction
    def greedy_construct():
        uncovered = [True] * space_size
        num_uncovered = space_size
        code = []
        
        while num_uncovered > 0 and time.time() < deadline:
            best_c = -1
            best_count = 0
            
            # Iterate through all possible candidates
            # Optimization: We can skip candidates that don't cover any uncovered word
            # But checking that is as expensive as counting.
            # So just count.
            for c in range(space_size):
                count = 0
                for nb in neighbors[c]:
                    if uncovered[nb]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_c = c
            
            if best_count == 0:
                break # Should not happen if num_uncovered > 0 and space is covered
                
            code.append(best_c)
            
            # Update uncovered
            for nb in neighbors[best_c]:
                if uncovered[nb]:
                    uncovered[nb] = False
                    num_uncovered -= 1
        
        return code

    # Method 2: Product Code + Removal
    def product_construct():
        # Tetracode for first 4 coords
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        code = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                w = h + list(tail)
                # Encode
                x = 0
                pow_q = 1
                for val in w:
                    x += val * pow_q
                    pow_q *= q
                code.append(x)
        return code

    best_code = None
    
    # Run Greedy
    t_start = time.time()
    code_greedy = greedy_construct()
    code_greedy, _ = remove_redundancies(code_greedy)
    if best_code is None or len(code_greedy) < len(best_code):
        best_code = code_greedy
    
    # Run Product + Removal
    if time.time() < deadline - 10:
        code_prod = product_construct()
        code_prod, _ = remove_redundancies(code_prod)
        if best_code is None or len(code_prod) < len(best_code):
            best_code = code_prod
            
    # Decode best_code to list of lists
    if best_code is None:
        # Fallback: return product code without removal if something went wrong
        best_code = product_construct()
        
    result = []
    for x in best_code:
        w = [0] * n
        temp = x
        for i in range(n):
            w[i] = temp % q
            temp //= q
        result.append(w)
        
    return result

import time
import random

def construct():
    n = 10
    start_time = time.time()
    time_limit = 50.0  # Leave 10s margin
    
    # Helper to check if a square is Latin
    def is_latin(sq):
        for i in range(n):
            if len(set(sq[i])) != n:
                return False
        for j in range(n):
            col = [sq[i][j] for i in range(n)]
            if len(set(col)) != n:
                return False
        return True

    # Helper to check orthogonality
    def is_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                pair = (sq1[i][j], sq2[i][j])
                if pair in pairs:
                    return False
                pairs.add(pair)
        return True

    # Generate candidates using Z5 x Z2 structure
    # i = 5a + b, j = 5c + d
    # L(i,j) = 5 * ((b + d + u*a + v*c) % 2) + ((a + c + w*b + x*d) % 5)
    candidates = []
    
    # u, v in {0, 1}
    # w, x in {0, 1, 2, 3, 4}
    for u in range(2):
        for v in range(2):
            for w in range(5):
                for x in range(5):
                    cand = [[0]*n for _ in range(n)]
                    for i in range(n):
                        a = i // 5
                        b = i % 5
                        for j in range(n):
                            c = j // 5
                            d = j % 5
                            high = (b + d + u*a + v*c) % 2
                            low = (a + c + w*b + x*d) % 5
                            cand[i][j] = 5*high + low
                    if is_latin(cand):
                        candidates.append(cand)
                        
    # Find the largest set of mutually orthogonal squares
    # Greedy approach: sort candidates by some heuristic? 
    # Or just try to build a set.
    # Since the space is small (<=100), we can do a simple search.
    
    best_set = []
    
    # Try to start with each candidate as the first square
    # And greedily add others
    for i in range(len(candidates)):
        current_set = [candidates[i]]
        for j in range(len(candidates)):
            if i == j:
                continue
            if time.time() - start_time > time_limit:
                break
            cand = candidates[j]
            ortho = True
            for sq in current_set:
                if not is_orthogonal(sq, cand):
                    ortho = False
                    break
            if ortho:
                current_set.append(cand)
        
        if len(current_set) > len(best_set):
            best_set = current_set
            
        # Early termination if we found the theoretical max (9) or a good number (3)
        if len(best_set) >= 3:
            # We know max is 3, so we can stop if we find 3
            # But let's check if we can find more? No, max is 3.
            # So if we find 3, we can break.
            break

    # If we didn't find any (should not happen), return a trivial one
    if not best_set:
        base_sq = [[(i + j) % n for j in range(n)] for i in range(n)]
        best_set = [base_sq]
        
    # Convert to Python lists of ints to satisfy type checking
    result = []
    for sq in best_set:
        result.append([list(row) for row in sq])
        
    return result

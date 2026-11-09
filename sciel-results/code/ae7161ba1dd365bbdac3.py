import time
import itertools

def construct():
    # Deadline guard: 90s wall-clock limit, use 85s as safety margin
    deadline = time.time() + 85.0
    
    n = 10
    
    # Base Latin square: L[i][j] = (i + j) % n
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Verify base is a valid Latin square
    def is_latin(sq):
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
        for j in range(n):
            col = [sq[i][j] for i in range(n)]
            if sorted(col) != list(range(n)):
                return False
        return True
    
    if not is_latin(base):
        return [base]  # Should not happen, but safety
    
    # Check orthogonality between two squares
    def is_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                pair = (sq1[i][j], sq2[i][j])
                if pair in pairs:
                    return False
                pairs.add(pair)
        return True
    
    # Try to find a second square orthogonal to base
    # Candidate: try all linear forms (a*i + b*j) % n where gcd(a,n)=1 and gcd(b,n)=1
    # This gives Latin squares that may or may not be orthogonal to base
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    candidates = []
    for a in range(1, n):
        for b in range(1, n):
            if gcd(a, n) == 1 and gcd(b, n) == 1:
                sq = [[(a * i + b * j) % n for j in range(n)] for i in range(n)]
                if is_latin(sq):
                    candidates.append(sq)
    
    # Try each candidate against base
    found_second = None
    for sq in candidates:
        if is_orthogonal(base, sq):
            found_second = sq
            break
    
    if found_second is not None:
        result = [base, found_second]
    else:
        result = [base]
    
    # Try to find a third square orthogonal to both base and found_second
    if len(result) == 2:
        third_candidates = []
        for a in range(1, n):
            for b in range(1, n):
                if gcd(a, n) == 1 and gcd(b, n) == 1:
                    sq = [[(a * i + b * j) % n for j in range(n)] for i in range(n)]
                    if is_latin(sq):
                        if is_orthogonal(base, sq) and is_orthogonal(result[1], sq):
                            third_candidates.append(sq)
        
        if third_candidates:
            result.append(third_candidates[0])
    
    # Ensure all elements are Python ints (not numpy or bool)
    final_result = []
    for sq in result:
        final_sq = []
        for row in sq:
            final_row = [int(x) for x in row]
            final_sq.append(final_row)
        final_result.append(final_sq)
    
    return final_result

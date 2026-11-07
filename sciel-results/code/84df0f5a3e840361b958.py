import itertools
import numpy as np

def construct():
    n = 10
    # Start with the standard cyclic Latin square
    squares = []
    L0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares.append(L0)
    
    # Try to add more squares using backtracking
    # We'll try squares of the form L[i][j] = (a*i + b*j) % n for various a, b
    # But this won't work for composite n in general. Let's try a different approach.
    
    # Try to find MOLS by searching
    # For each new square, we need it to be a Latin square and orthogonal to all existing ones
    
    def is_latin_square(sq):
        for i in range(n):
            if set(sq[i]) != set(range(n)):
                return False
        for j in range(n):
            col = [sq[i][j] for i in range(n)]
            if set(col) != set(range(n)):
                return False
        return True
    
    def is_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                pair = (sq1[i][j], sq2[i][j])
                if pair in pairs:
                    return False
                pairs.add(pair)
        return len(pairs) == n*n
    
    # Try to add squares using a heuristic search
    # We'll generate candidate squares and check
    
    # First, try linear squares with different coefficients
    # L(i,j) = (a*i + b*j) % n, but this only works if gcd conditions are met
    # For n=10, let's try various a, b where gcd(a,10) and gcd(b,10) are such that it's a Latin square
    
    # A Latin square from L(i,j) = (a*i + b*j) % n requires that the mapping (i,j) -> (a*i+b*j)%n is bijective in each row and column
    # In row i, j -> (a*i + b*j) % n is bijective iff gcd(b, n) = 1
    # In column j, i -> (a*i + b*j) % n is bijective iff gcd(a, n) = 1
    
    # So we need gcd(a,10)=1 and gcd(b,10)=1, so a,b in {1,3,7,9}
    
    candidate_squares = []
    for a in [1, 3, 7, 9]:
        for b in [1, 3, 7, 9]:
            sq = [[(a*i + b*j) % n for j in range(n)] for i in range(n)]
            candidate_squares.append((a, b, sq))
    
    # Check which of these are orthogonal to L0
    valid_candidates = []
    for a, b, sq in candidate_squares:
        if is_orthogonal(L0, sq):
            valid_candidates.append((a, b, sq))
    
    # Now try to find a set of mutually orthogonal squares from valid_candidates plus L0
    # Use backtracking
    
    def backtrack(squares_so_far, remaining_candidates, index):
        if not remaining_candidates:
            return squares_so_far
        
        for idx, (a, b, sq) in enumerate(remaining_candidates):
            # Check if sq is orthogonal to all in squares_so_far
            orthogonal = True
            for existing in squares_so_far:
                if not is_orthogonal(sq, existing):
                    orthogonal = False
                    break
            if orthogonal:
                new_squares = squares_so_far + [sq]
                new_remaining = remaining_candidates[:idx] + remaining_candidates[idx+1:]
                result = backtrack(new_squares, new_remaining, index+1)
                if result:
                    return result
        
        return squares_so_far
    
    # Try to find the maximum set
    # Start with L0 and try to add from valid_candidates
    remaining = valid_candidates
    best = [L0]
    
    # Greedy approach: try to add as many as possible
    current = [L0]
    used = set()
    for (a, b, sq) in valid_candidates:
        orthogonal = True
        for existing in current:
            if not is_orthogonal(sq, existing):
                orthogonal = False
                break
        if orthogonal:
            current.append(sq)
            used.add((a, b))
    
    # If we have at least 2 squares, return them
    if len(current) >= 1:
        return current
    
    # Fallback: just return the one square
    return [L0]

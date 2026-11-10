import random

def construct():
    """
    Return a polynomial with exactly 12 nonzero terms whose square has
    as few nonzero terms as possible.
    
    Searches for a polynomial of the form P(x) = A(x) - x^6 A(x)
    where A(x) has 6 nonzero terms, to maximize cancellations in P(x)^2.
    """
    # Initial best candidate (the seed solution)
    best_poly = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
    best_score = 21
    
    def count_nonzeros(poly):
        return sum(1 for c in poly if c != 0)
    
    def square_nonzeros(A):
        # A is list of 6 coeffs
        # P = A - x^6 A
        # P^2 = A^2 - 2 x^6 A^2 + x^12 A^2
        # Calculate A^2
        A2 = [0]*11
        for i in range(6):
            for j in range(6):
                if i+j < 11:
                    A2[i+j] += A[i]*A[j]
        
        C = [0]*23
        for k in range(11):
            C[k] += A2[k]
            if k+6 < 23:
                C[k+6] -= 2*A2[k]
            if k+12 < 23:
                C[k+12] += A2[k]
        
        return count_nonzeros(C)

    # Random search for better A
    # Coeffs in [-5, 5] to keep P coeffs small
    for _ in range(50000):
        A = [random.randint(-5, 5) for _ in range(6)]
        if 0 in A:
            continue
        score = square_nonzeros(A)
        if score < best_score:
            best_score = score
            best_poly = A + [-x for x in A]
            if best_score <= 19:
                break

    # Verify exactly 12 nonzero terms
    nonzero_count = sum(1 for c in best_poly if c != 0)
    assert nonzero_count == 12, f"Expected 12 nonzero terms, got {nonzero_count}"
    
    # Verify all entries are in [-64, 64]
    assert all(-64 <= c <= 64 for c in best_poly), "Coefficient out of range"
    
    # Verify length <= 41
    assert len(best_poly) <= 41, "Polynomial too long"
    
    # Verify nonzero final coefficient
    assert best_poly[-1] != 0, "Final coefficient must be nonzero"
    
    return best_poly

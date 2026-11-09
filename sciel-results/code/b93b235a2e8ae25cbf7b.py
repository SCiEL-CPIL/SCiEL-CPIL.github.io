"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill

    def count_nonzero_square(coeffs):
        """Calculate the number of non-zero coefficients in the square of the polynomial."""
        n = len(coeffs)
        square_coeffs = [0] * (2 * n - 1)
        for i in range(n):
            ai = coeffs[i]
            if ai == 0:
                continue
            for j in range(i, n):
                aj = coeffs[j]
                if aj == 0:
                    continue
                k = i + j
                prod = ai * aj
                if i == j:
                    square_coeffs[k] += prod
                else:
                    square_coeffs[k] += 2 * prod
        return sum(1 for c in square_coeffs if c != 0)

    # The current program searches over the full 12-dimensional space of coefficients.
    # This new strategy uses a STRUCTURAL DECOMPOSITION: P(x) = A(x) * B(x).
    # We enforce that P has exactly 12 nonzero terms by choosing A and B such that
    # their convolution has exactly 12 nonzero entries.
    # We then optimize the coefficients of A and B to minimize the sparsity of P^2.
    
    # A key insight: If P(x) is a product of two polynomials, its structure is constrained.
    # We can enumerate small polynomials A and B, compute P = A*B, check if P has exactly
    # 12 nonzero terms, and then evaluate the sparsity of P^2.
    # This reduces the search space significantly because we are only exploring structured P.

    best_count = 23
    best_poly = None

    # Helper to check if a polynomial has exactly 12 nonzero terms
    def has_12_nonzero(poly):
        return sum(1 for c in poly if c != 0) == 12 and len(poly) <= 41 and poly[-1] != 0

    # Helper to trim trailing zeros (but keep at least one term)
    def trim_poly(poly):
        # Ensure nonzero final coefficient as per contract
        # The contract says "nonzero final coefficient", which means the last element of the list is nonzero.
        # But also "exactly 12 nonzero terms".
        # If we trim, we must ensure the last element is nonzero.
        # Actually, the contract says "Return coefficient list in ascending degree, length<=41, nonzero final coefficient".
        # This implies the list length is the degree+1, and the last element is nonzero.
        # So we should not have trailing zeros.
        while len(poly) > 1 and poly[-1] == 0:
            poly.pop()
        return poly

    # Phase 1: Enumerate small polynomials A and B
    # Let A have degree da, B have degree db.
    # P = A*B has degree da+db.
    # We want P to have exactly 12 nonzero terms.
    
    # Try small degrees for A and B
    # For example, A of degree 3, B of degree 5 -> P of degree 8 (9 terms max, too few)
    # A of degree 5, B of degree 6 -> P of degree 11 (12 terms max, good)
    # A of degree 4, B of degree 7 -> P of degree 11 (12 terms max, good)
    
    # Let's try A of degree 5, B of degree 6.
    # Coefficients in {-1, 0, 1} for A and B.
    # 3^6 * 3^7 = 729 * 2187 = ~1.6 million. Might be too slow in Python.
    # Let's restrict to {-1, 1} first.
    # 2^6 * 2^7 = 64 * 128 = 8192. Very fast.
    
    # Phase 1a: A and B with coefficients in {-1, 1}
    for da in range(2, 6):
        for db in range(2, 7):
            if da + db > 15:  # Keep degree reasonable
                continue
            # Enumerate A
            for a_mask in range(1 << (da + 1)):
                if time.time() > deadline:
                    return best_poly if best_poly else [1]*12
                A = [1 if (a_mask >> i) & 1 else -1 for i in range(da + 1)]
                # Enumerate B
                for b_mask in range(1 << (db + 1)):
                    B = [1 if (b_mask >> i) & 1 else -1 for i in range(db + 1)]
                    # Compute P = A * B
                    P = np.convolve(A, B)
                    P = list(P)
                    if has_12_nonzero(P):
                        c = count_nonzero_square(P)
                        if c < best_count:
                            best_count = c
                            best_poly = P[:]
                            if best_count <= 5:
                                return best_poly

    # Phase 1b: A and B with coefficients in {-1, 0, 1}
    # This is 3^(da+1) * 3^(db+1).
    # For da=5, db=6: 3^6 * 3^7 = 729 * 2187 = 1,593,921.
    # Each convolution is O(da*db) ~ 30 ops.
    # Total ~ 50 million ops. Might be slow in Python.
    # Let's try a subset or smaller degrees.
    
    # Let's try da=3, db=8 -> P degree 11 (12 terms)
    # 3^4 * 3^9 = 81 * 19683 = 1,594,323. Similar cost.
    
    # Instead, let's use a smarter search: fix A, and for each A, find B that makes P sparse.
    # Or, use local search on A and B.
    
    # Phase 2: Local search on A and B
    # Start from the best P found in Phase 1a, or a random structured P.
    # If best_poly is found, try to factor it? No, we generated it from A*B.
    # Let's keep the A, B that gave the best P.
    
    # Re-run Phase 1a but store A, B
    best_A = None
    best_B = None
    
    for da in range(2, 6):
        for db in range(2, 7):
            if da + db > 15:
                continue
            for a_mask in range(1 << (da + 1)):
                if time.time() > deadline:
                    if best_poly:
                        return best_poly
                    else:
                        return [1]*12
                A = [1 if (a_mask >> i) & 1 else -1 for i in range(da + 1)]
                for b_mask in range(1 << (db + 1)):
                    B = [1 if (b_mask >> i) & 1 else -1 for i in range(db + 1)]
                    P = list(np.convolve(A, B))
                    if has_12_nonzero(P):
                        c = count_nonzero_square(P)
                        if c < best_count:
                            best_count = c
                            best_poly = P[:]
                            best_A = A[:]
                            best_B = B[:]
                            if best_count <= 5:
                                return best_poly

    # Phase 3: Local search on A and B coefficients
    if best_A and best_B:
        A = best_A[:]
        B = best_B[:]
        current_count = best_count
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try changing one coefficient in A
            for i in range(len(A)):
                for new_val in [-2, -1, 1, 2]:
                    if new_val == A[i]:
                        continue
                    old_val = A[i]
                    A[i] = new_val
                    P = list(np.convolve(A, B))
                    if has_12_nonzero(P):
                        new_count = count_nonzero_square(P)
                        if new_count < current_count:
                            current_count = new_count
                            best_count = current_count
                            best_poly = P[:]
                            improved = True
                        else:
                            A[i] = old_val
                    else:
                        A[i] = old_val
            # Try changing one coefficient in B
            for i in range(len(B)):
                for new_val in [-2, -1, 1, 2]:
                    if new_val == B[i]:
                        continue
                    old_val = B[i]
                    B[i] = new_val
                    P = list(np.convolve(A, B))
                    if has_12_nonzero(P):
                        new_count = count_nonzero_square(P)
                        if new_count < current_count:
                            current_count = new_count
                            best_count = current_count
                            best_poly = P[:]
                            improved = True
                        else:
                            B[i] = old_val
                    else:
                        B[i] = old_val

    # Phase 4: Random structured search
    # Random A and B with small coefficients, check if P has 12 nonzero terms.
    for r in range(5000):
        if time.time() > deadline:
            break
        da = random.randint(2, 5)
        db = random.randint(2, 6)
        A = [random.choice([-2, -1, 1, 2]) for _ in range(da + 1)]
        B = [random.choice([-2, -1, 1, 2]) for _ in range(db + 1)]
        P = list(np.convolve(A, B))
        if has_12_nonzero(P):
            c = count_nonzero_square(P)
            if c < best_count:
                best_count = c
                best_poly = P[:]
                if best_count <= 5:
                    break

    if best_poly is None:
        # Fallback: a known good polynomial
        # P(x) = 1 + x + x^2 + ... + x^11
        # Square has 23 terms.
        # Let's try a random one with 12 nonzero terms
        random.seed(42)
        for _ in range(1000):
            P = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            if has_12_nonzero(P):
                c = count_nonzero_square(P)
                if c < best_count:
                    best_count = c
                    best_poly = P[:]

    if best_poly is None:
        best_poly = [1] * 12

    # Ensure constraints
    # 1. Length <= 41
    # 2. Nonzero final coefficient
    # 3. Entries in [-64, 64]
    # 4. Exactly 12 nonzero terms
    
    # Trim trailing zeros
    best_poly = trim_poly(best_poly)
    
    # Check if we have exactly 12 nonzero terms
    if sum(1 for c in best_poly if c != 0) != 12:
        # Fallback to a valid polynomial
        # Find one by search
        for _ in range(1000):
            P = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            if has_12_nonzero(P):
                best_poly = P
                break
    
    return best_poly

# EVOLVE-BLOCK-END

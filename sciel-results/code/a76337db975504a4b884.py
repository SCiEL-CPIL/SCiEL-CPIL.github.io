"""Fresh design: randomized integer rank search for 2x3 x 3x8 multiplication tensor."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]
N, M, P = DIMENSIONS
NM, MP, PN = N * M, M * P, P * N
SCHOOLBOOK_RANK = N * M * P  # 48


def _schoolbook():
    rank = SCHOOLBOOK_RANK
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][c] = 1
                V[j * P + k][c] = 1
                W[k * N + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _build_tensor_matrix():
    # T as (NM*MP) x PN matrix: rows (i,j,k) flattened, cols (k,i)
    # We want: for each (i,j,k): sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    # This is nonlinear. Instead, treat as: given U and V, solve for W.
    # For each (k,i): sum_j sum_c U[i,j,c]*V[j,k,c] * W[k,i,c] = 1
    # Let A_{(k,i),c} = sum_j U[i,j,c]*V[j,k,c]. Then A * W_col = 1 for each (k,i).
    pass


def _try_rank(r, rng, deadline):
    """Try to find integer rank-r decomposition. Returns dict or None."""
    # Strategy: randomly assign U and V with small integers, then solve for W.
    # A_{(k,i),c} = sum_j U[i*m+j, c] * V[j*p+k, c]
    # We need A * w_{k,i} = 1 for all (k,i), where w_{k,i} is the c-th row of W for index (k*n+i).
    # W has PN rows, each of length r. For each (k,i), we solve A_{(k,i),:} dot W[(k*n+i),:] = 1.
    # But W[(k*n+i),:] must be integer.
    # Actually W is shared: W[k*n+i, c] appears in the equation for (k,i).
    # So for each (k,i): sum_c A_{(k,i),c} * W[k*n+i, c] = 1.
    # This is a linear equation in W[k*n+i, :]. We need integer solutions.
    # A_{(k,i),c} = sum_j U[i*m+j, c] * V[j*p+k, c]
    # 
    # Let's build the full system. For each (k,i), we have one equation:
    # sum_c A_{(k,i),c} * W[k*n+i, c] = 1
    # 
    # So we have PN equations, each involving r unknowns (the W row for that (k,i)).
    # These are independent! Each (k,i) has its own equation.
    # So we need: for each (k,i), find integer vector w such that A_{(k,i),:} dot w = 1.
    # This is possible iff gcd(A_{(k,i),:}) divides 1, i.e., gcd = 1.
    # Then we can find a particular solution via extended Euclidean / lattice.
    
    # Random U and V
    max_val = 3
    U = [[rng.randint(-max_val, max_val) for _ in range(r)] for _ in range(NM)]
    V = [[rng.randint(-max_val, max_val) for _ in range(r)] for _ in range(MP)]
    
    # Build A for each (k,i)
    W = [[0] * r for _ in range(PN)]
    for k in range(P):
        for i in range(N):
            # A_c = sum_j U[i*M+j, c] * V[j*P+k, c]
            A = [0] * r
            for j in range(M):
                for c in range(r):
                    A[c] += U[i * M + j][c] * V[j * P + k][c]
            # Find integer w with A dot w = 1
            # Check gcd
            from math import gcd
            g = 0
            for a in A:
                g = gcd(g, a)
            if g != 1:
                return None
            # Find one solution via extended gcd chain
            # Simple approach: use sympy-like or just try small vectors
            # Use lattice basis reduction or just brute force for small r
            # For r up to ~30, brute force is too slow. Use extended Euclidean.
            # Actually, we can find a solution by: pick the index with smallest |A|, use ext_gcd
            # Better: use the fact that we can set all but one to 0 and solve A[idx]*w[idx]=1 -> need A[idx]=±1
            # That's too restrictive. Let's use a proper integer linear solve.
            
            # Use a simple method: find a particular solution using the extended Euclidean algorithm
            # for the system A dot w = 1. This is a single linear Diophantine equation.
            # Solution exists iff gcd(A) | 1. We can find one by:
            # 1. Compute the gcd and the coefficients via extended Euclidean on the list.
            
            # Extended Euclidean for multiple numbers to find coefficients giving gcd
            # We want x_c such that sum A_c * x_c = gcd = 1
            # Use the standard multi-extended Euclidean:
            
            # Start with the first two
            if r == 0:
                return None
            # Use a simple approach: since we just need ONE solution, and r is small,
            # use the fact that we can reduce: find two indices with coprime values...
            # Actually simplest: use numpy.linalg with rational? No, we need integers.
            
            # Let's use a straightforward extended GCD for the whole list:
            def ext_gcd_multi(coeffs):
                """Returns (g, x) such that sum coeffs[i]*x[i] = g = gcd(coeffs)"""
                if len(coeffs) == 1:
                    g = abs(coeffs[0])
                    x = [0]
                    if g == 0:
                        return 0, x
                    x[0] = 1 if coeffs[0] > 0 else -1
                    return g, x
                # Combine first two
                a, b = coeffs[0], coeffs[1]
                def egcd(a, b):
                    if b == 0:
                        return (abs(a), 1 if a > 0 else -1, 0)
                    g, x1, y1 = egcd(b, a % b)
                    return (g, y1, x1 - (a // b) * y1)
                g1, x1, y1 = egcd(a, b)
                # Now combine g1 with rest
                rest = coeffs[2:]
                g2, x2 = ext_gcd_multi([g1] + rest)
                # x2[0] * g1 + sum x2[1:]*rest = g2
                # = x2[0]*(x1*a + y1*b) + sum x2[1:]*rest
                x = [0] * len(coeffs)
                x[0] = x2[0] * x1
                x[1] = x2[0] * y1
                for i in range(len(rest)):
                    x[2 + i] = x2[1 + i]
                return g2, x
            
            g, x = ext_gcd_multi(A)
            if g != 1:
                return None
            W[k * N + i] = x
    
    # Verify
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0
                for c in range(r):
                    s += U[i * M + j][c] * V[j * P + k][c] * W[k * N + i][c]
                if s != 1:
                    return None
    return {"U": U, "V": V, "W": W}


def construct():
    start = time.time()
    deadline = start + 120.0
    rng = random.Random(42)
    
    # Try decreasing ranks
    for r in range(1, SCHOOLBOOK_RANK + 1):
        if time.time() > deadline - 30:
            break
        # Try several random attempts
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
            res = _try_rank(r, rng, deadline)
            if res is not None:
                return res
    
    # Fallback to schoolbook
    return _schoolbook()

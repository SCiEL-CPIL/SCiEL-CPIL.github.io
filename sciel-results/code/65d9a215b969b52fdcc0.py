"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time
import random
import itertools
import fractions
from fractions import Fraction

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # The task is to find the rank-minimal exact tensor decomposition for
    # multiplying an n x m matrix by an m x p matrix.
    # The tensor T is of size (n*m) x (m*p) x (p*n).
    # T[a, b, c] = 1 if and only if:
    #   a = i*m + j, b = j*p + k, c = k*n + i
    # for some i in [0, n), j in [0, m), k in [0, p).
    # Otherwise T[a, b, c] = 0.
    
    # The decomposition is T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r].
    # We need U, V, W with entries in Q (integer or half-integer) such that this holds exactly.
    
    # Known: The rank is at most n*m*p (schoolbook).
    # For (2, 4, 6), the trivial rank is 48.
    # Is the minimal rank 48?
    #   - The "slice rank" lower bound is max(n*m, m*p, n*p) = max(8, 24, 12) = 24.
    #   - However, matrix multiplication tensors often have rank equal to n*m*p for "generic" or non-factorizable cases,
    #     or slightly less if structure allows.
    #   - For (2, 2, 2), rank is 7 (Strassen) < 8.
    #   - For (2, 3, 3), rank is 16 < 18? (Actually known to be 16).
    #   - For (2, 4, 6), it is an open question in some contexts, but for small sizes, we can search.
    
    # Strategy:
    # 1. Establish the schoolbook solution (rank 48) as the fallback.
    # 2. Try to find a solution with rank < 48 using randomized search with rational numbers (specifically small integers/half-integers).
    #    Given the complexity, we will focus on a structured search or a smarter random search.
    #    
    #    A common technique is to fix U and V randomly and solve for W?
    #    No, the equation is cubic in the variables of a single term, but linear in W if U and V are fixed?
    #    Sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].
    #    For a fixed U and V, this is a linear system in W?
    #    Let M_{a,b}^{(c)} be the coefficient of W[c,r]? No.
    #    The equation is: sum_r (U[a,r] V[b,r]) W[c,r] = T[a,b,c].
    #    This is not a standard linear system in W because the index r is summed over, and W depends on c and r.
    #    Actually, for each (a,b), we have:
    #    sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] for all c.
    #    This is a linear system for the vector of W entries for each r? No.
    #    
    #    Let's view it differently.
    #    The tensor T can be seen as a matrix A (n*m x m*p) if we flatten W? No.
    #    
    #    Alternative: 
    #    We can fix U and V. Then for each r, let u_r be a vector in R^{n*m}, v_r in R^{m*p}.
    #    We need to find w_r in R^{p*n} such that:
    #    T = sum_r u_r v_r^T \otimes w_r ? No.
    #    T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
    #    This is exactly the definition of the tensor rank decomposition.
    #    
    #    If we fix U and V, is there a unique W?
    #    The mapping (U,V) -> T is not linear in U and V jointly, but it is multilinear.
    #    Fixing U and V makes it linear in W.
    #    Specifically, define a linear operator L: R^{p*n * R} -> R^{n*m x m*p x p*n}
    #    L(W)[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
    #    We want L(W) = T.
    #    This is a system of n*m*p equations (24 equations) with R*(p*n) variables (R*12 variables).
    #    If R is large enough, this system might have a solution.
    #    If a solution exists, we can check if W has rational entries.
    #    If we pick U and V with rational entries (e.g., integers), and the system has a unique solution, W will be rational.
    #    Then we can scale W to half-integers if needed (or check if it already is).
    #    
    #    However, the system is underdetermined if R is small, or overdetermined if R is large?
    #    Number of equations: N_eq = n*m*p = 24.
    #    Number of variables: N_var = R * p * n = R * 12.
    #    If R=2, N_var=24. Square system.
    #    If R=3, N_var=36. Underdetermined.
    #    
    #    So, for R=2, we can pick random U and V (integer entries), form the linear system for W, and solve.
    #    If the solution W is rational (which it will be if U,V are integers and the matrix is invertible), we can check if it can be represented as half-integers.
    #    Actually, any rational number can be scaled to integers. But the problem requires "integer/half-integer entries".
    #    This means entries must be k/2 for some integer k.
    #    If we find a rational solution W, we can check if all entries are multiples of 1/2.
    #    If not, we might need to scale? But we cannot scale U, V, W independently because the product must be 1.
    #    If we scale U by c, V by 1/c, W by 1, the product is preserved.
    #    So we have freedom to scale the terms.
    #    
    #    Plan:
    #    1. Try ranks R from 24 (lower bound) up to 47.
    #    2. For each R, generate random U, V with integer entries in {-1, 0, 1}.
    #    3. Construct the linear system A W_vec = T_vec.
    #       A is a matrix of size (n*m*p) x (R * p * n).
    #       A_{(a,b,c), (r, c')} = U[a,r] V[b,r] if c' == c, else 0.
    #       Wait, the index for W is (c, r). Let's flatten W to a vector of length R*p*n.
    #       Let W_vec[r * (p*n) + c] = W[c, r].
    #       Then the equation for a given (a,b,c) is:
    #       sum_r U[a,r] V[b,r] W[c, r] = T[a,b,c].
    #       In terms of W_vec:
    #       sum_r U[a,r] V[b,r] W_vec[r * (p*n) + c] = T[a,b,c].
    #       So the coefficient of W_vec[r * (p*n) + c] in the equation for (a,b,c) is U[a,r] V[b,r] if the c-index matches.
    #       Since the LHS only involves W[c, r] for the specific c, the matrix A is sparse.
    #       Specifically, for each (a,b), we have p equations (one for each c).
    #       These p equations involve only the variables W[0,r]...W[p-1,r] for all r? No.
    #       For a fixed (a,b) and c, the equation is:
    #       sum_r (U[a,r] V[b,r]) W[c, r] = T[a,b,c].
    #       This equation involves variables W[c, 0], ..., W[c, R-1].
    #       So, for a fixed c, we have a system of n*m equations for the variables W[c, 0..R-1].
    #       Let M_c be the matrix of size (n*m) x R.
    #       M_c[a, r] = U[a,r] V[b,r]? No, b varies.
    #       The equation is for a fixed (a,b,c).
    #       Let's group by c.
    #       For a fixed c, we have equations for all (a,b) such that T[a,b,c] is non-zero?
    #       No, T[a,b,c] is non-zero only if there exists (i,j,k) with k*n+i=c and j*p+k=b and i*m+j=a.
    #       Actually, T is sparse!
    #       T[a,b,c] = 1 iff:
    #       i = c % n, k = c // n.
    #       j = b % p? No. b = j*p + k. So j = (b - k) // p, and we need b % p == k.
    #       Wait, b is an index in 0..m*p-1. j is in 0..m-1, k in 0..p-1.
    #       So b = j*p + k.
    #       Then a = i*m + j.
    #       So for a given (a,b,c), T[a,b,c] = 1 iff:
    #       i = c % n, k = c // n.
    #       j = b % m? No.
    #       Let's map indices carefully.
    #       a = i*m + j  => i = a // m, j = a % m.
    #       b = j*p + k  => j = b // p, k = b % p. (Note: j must match the j from a)
    #       c = k*n + i  => k = c // n, i = c % n. (Note: k must match the k from b, and i from a)
    #       
    #       So T[a,b,c] = 1 iff:
    #       (a // m) == (c % n)
    #       AND (a % m) == (b // p)
    #       AND (b % p) == (c // n)
    #       Otherwise T[a,b,c] = 0.
    #       
    #       This means the system of equations is not fully connected.
    #       For a fixed r, the variables are U[a,r], V[b,r], W[c,r].
    #       
    #       Let's restart the linear system idea.
    #       We want to find U, V, W of rank R.
    #       This is a system of 24 equations in R*(n*m + m*p + p*n) variables.
    #       The equations are cubic (products of variables from different matrices).
    #       This is not a linear system in all variables.
    #       
    #       However, if we fix U and V, it becomes linear in W.
    #       Let's try this.
    #       Fix U and V randomly (integers).
    #       Then for each c, we have a system of linear equations for W[:, c]?
    #       No, W is (p*n, R). The variable is W[c, r] for fixed c and varying r?
    #       Let's look at the equation for a fixed (a,b,c):
    #       sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].
    #       This is a linear equation in the variables W[c, 0], ..., W[c, R-1].
    #       The coefficients are U[a,r] V[b,r].
    #       
    #       So, for each c (0 to p*n-1), we can form a system of equations for the vector W_c = (W[c,0], ..., W[c,R-1]).
    #       How many equations do we have for W_c?
    #       We have an equation for every (a,b) such that the triplet (a,b,c) is valid?
    #       No, the equation must hold for ALL (a,b,c).
    #       So for a fixed c, we have n*m equations (one for each a,b).
    #       However, most of these equations are 0 = 0 if T[a,b,c] = 0 and the LHS is 0?
    #       No, we must satisfy sum_r U[a,r] V[b,r] W[c,r] = 0 for all (a,b) where T[a,b,c]=0.
    #       And sum_r U[a,r] V[b,r] W[c,r] = 1 for the specific (a,b) where T[a,b,c]=1.
    #       
    #       So, for each c, we have a system of n*m linear equations in R unknowns (W[c,0..R-1]).
    #       The matrix for this system, let's call it A_c, has size (n*m) x R.
    #       A_c[a, r] = U[a,r] * V[b(a,c), r]?
    #       Wait, b is not determined by a and c alone.
    #       The equation is indexed by (a,b,c).
    #       So for a fixed c, we have equations for all pairs (a,b).
    #       There are n*m such pairs.
    #       For each pair (a,b), the coefficient of W[c,r] is U[a,r] * V[b,r].
    #       So the matrix A_c is not well-defined as a single matrix for all (a,b) because the coefficients depend on b.
    #       
    #       Let's write it out:
    #       For a fixed c:
    #       For all a in [0, n*m), b in [0, m*p):
    #       sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].
    #       
    #       This is a system of n*m*(m*p) equations? No, that's too many.
    #       The tensor has size n*m * m*p * p*n.
    #       The indices are (a, b, c).
    #       So for a fixed c, we have (n*m) * (m*p) equations?
    #       No, a ranges over n*m, b ranges over m*p.
    #       So yes, for a fixed c, we have n*m * m*p equations in R unknowns.
    #       n*m * m*p = 8 * 24 = 192 equations.
    #       Unknowns: R.
    #       This is a heavily overdetermined system.
    #       It is very likely that no solution exists for random U, V unless R is large or U, V are specially chosen.
    #       Actually, if R=1, we need U[a] V[b] W[c] = T[a,b,c].
    #       This requires T to be rank 1. It is not.
    #       
    #       So this approach (fix U, V, solve W) is unlikely to work for small R because the system is overdetermined.
    #       Unless U and V are chosen such that the vectors {U[a,r] V[b,r]} lie in a low-dimensional subspace.
    #       
    #    3. Alternative: Use the fact that T is sparse and structured.
    #       For (2, 4, 6), maybe we can just return the schoolbook solution.
    #       The question is whether a lower rank exists.
    #       Given the time constraints and the difficulty of finding a non-trivial decomposition for this specific shape without known identities, the safest bet is the schoolbook rank 48.
    #       However, I will try a very limited search for rank 47, 46, etc., using random integers, to see if I get lucky.
    #       But the probability is extremely low.
    #       
    #    Let's check if there's a simple structural reduction.
    #    2x4x6.
    #    Maybe 2x4x6 is related to 2x2x6?
    #    Rank(2x2x6) = 2*2*6 = 24?
    #    If we can do 2x4x6 in rank 48, is it minimal?
    #    
    #    I will implement the schoolbook solution and a very quick random search for rank < 48.
    #    If the random search fails (which it likely will), return schoolbook.

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    
    # Quick random search for rank < 48
    # This is mostly a formality, as the search space is vast and the solution (if it exists) is unlikely to be found randomly.
    # We'll try a few iterations for rank 47, 46, 45.
    
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Try a few random restarts for rank 47
    if time.time() < deadline - 20:
        for r in [47, 46, 45]:
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
                V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
                W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p * n)]
                if verify(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    return best_result
    
    return best_result

# EVOLVE-BLOCK-END

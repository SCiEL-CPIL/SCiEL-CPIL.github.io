import time
import numpy as np
from fractions import Fraction
from itertools import product as iproduct

DIMENSIONS = [2, 3, 6]
n, m, p = DIMENSIONS
NM = n * m          # 6
MP = m * p          # 18
PN = p * n          # 12
TOTAL = NM * MP * PN  # 1296

# Multiplication tensor in the trace convention W[k*n + i, r], U[i*m + j, c], V[j*p + k, c]
# Tensor T[a, b, c] = 1 if (a,b,c) corresponds to A[i,j]*B[j,k] -> C[i,k]
T = np.zeros((NM, MP, PN), dtype=np.int64)
for i in range(n):
    for j in range(m):
        for k in range(p):
            a = i * m + j
            b = j * p + k
            c = k * n + i
            T[a, b, c] = 1
T_flat = T.reshape(-1)

# Candidate gates: monomials of degree 1 (rank-1 with single A/B factor)
# A factor: choose (i, j)
# B factor: choose (j, k)
# Output: choose (i, k)
# Gate tensor has 1 at (i*m+j, j*p+k, k*n+i)
# This is exactly the schoolbook set, but we can also add "shifted" gates.

# Actually for rank-1 decomposition, each rank-1 term is u⊗v⊗w where u, v, w are vectors.
# The simplest gates are the schoolbook ones. But we can also use gates where
# u has a single 1, v has a single 1, w has a single 1 - that's the schoolbook.
# To get lower rank, we need gates where u, v, w have multiple non-zero entries.

# Strategy: enumerate gates where u is a unit vector (single A entry), v is a unit vector (single B entry),
# and w is a unit vector (single output entry). That's schoolbook.
# For lower rank, allow u to be a unit vector, v to be a unit vector, and w to be a sum of unit vectors,
# OR u a sum, v a unit, w a unit, etc.

# Let's enumerate gates of the form: u = e_a, v = e_b, w = sum of some e_c's
# The gate tensor is e_a ⊗ e_b ⊗ w, which has non-zeros at (a, b, c) for each c in support(w).

# For the multiplication tensor, T[a,b,c] = 1 only when a=i*m+j, b=j*p+k, c=k*n+i for the same (i,j,k).
# So for fixed (a,b) = (i*m+j, j*p+k), the output must be c = k*n+i.
# This means the gate with u=e_a, v=e_b, w=e_c where c=k*n+i is the only "clean" gate.

# To reduce rank, we need to combine multiple (a,b,c) into one gate.
# Idea: for a fixed j (the shared index), we can have a gate where
# u has 1's at all a = i*m+j for i=0..n-1,
# v has 1's at all b = j*p+k for k=0..p-1,
# w has 1's at all c = k*n+i for i=0..n-1, k=0..p-1.
# But that gate's tensor would have non-zeros at ALL (a,b,c) combinations in that block,
# not just the diagonal ones. So it doesn't match T.

# The tensor T is a permutation-like tensor (each (a,b) maps to exactly one c, and each c maps from exactly one (a,b) pair... let's check).
# Actually T[a,b,c] is 1 at exactly n*m*p = 36 positions.
# For rank-1 decomposition, we need to express T as sum of rank-1 tensors.

# Lower bound: rank >= max(n*m, m*p, p*n) = max(6, 18, 12) = 18? No, that's not a valid lower bound.
# The rank of a tensor is at least the dimension of any fiber. But that gives rank >= 1 for each fiber.
# A better lower bound: the rank is at least the max over all slices of the matrix rank of that slice.
# Slice fixing a: matrix of size MP x PN. For a = i*m+j, the slice has 1's at (j*p+k, k*n+i) for k=0..p-1.
# That's a p x (p*n) matrix with 1's in a specific pattern. Its rank is p (since the p rows are independent).
# So rank >= p = 6.
# Slice fixing b: matrix of size NM x PN. For b = j*p+k, the slice has 1's at (i*m+j, k*n+i) for i=0..n-1.
# That's an (n*m) x (p*n) matrix. The non-zero entries are at rows i*m+j and columns k*n+i.
# For fixed j,k, the entries are at (i*m+j, k*n+i) for i=0..n-1. These n entries form a permutation pattern.
# The rank of this slice: the n non-zero rows are independent, so rank >= n = 2.
# Slice fixing c: matrix of size NM x MP. For c = k*n+i, the slice has 1 at (i*m+j, j*p+k) for j=0..m-1.
# That's m entries, one per j. These are in different rows and columns, so rank = m = 3.

# So the maximum slice rank is max(6, 2, 3) = 6. This gives rank >= 6, which is weak.

# Let's try to find a decomposition of rank 18 (schoolbook) and see if we can do better.
# Actually, let's try to use the structure: the tensor is a "permutation tensor" in a sense.
# We can try rank = m*p = 18, which is the schoolbook. Can we do less?

# Let me try a different approach: use the fact that T is a 3-tensor and try to find a low-rank decomposition
# by solving a system of equations.

# For rank r, we need U (NM x r), V (MP x r), W (PN x r) such that
# sum_c U[a,c]*V[b,c]*W[c_out,c] = T[a,b,c_out] for all a,b,c_out.

# This is a system of NM*MP*PN = 1296 equations in r*(NM+MP+PN) = r*36 unknowns.
# For r=6, that's 216 unknowns, 1296 equations. Likely no solution.
# For r=18, that's 648 unknowns, 1296 equations. Might have a solution.

# Let's try to find a solution for small r using a greedy or random approach.

def verify(U, V, W):
    """Verify that U, V, W exactly decompose the tensor."""
    rank = U.shape[1]
    # Reconstruct tensor
    T_rec = np.zeros((NM, MP, PN), dtype=np.int64)
    for c in range(rank):
        T_rec += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
    return np.array_equal(T_rec, T)

def try_rank(r):
    """Try to find a rank-r decomposition using integer coefficients."""
    # Use a structured approach: each rank-1 term corresponds to a gate.
    # We'll enumerate all possible "simple" gates and try to combine them.
    
    # For now, return the schoolbook if r >= n*m*p
    if r >= n * m * p:
        U = np.zeros((NM, r), dtype=np.int64)
        V = np.zeros((MP, r), dtype=np.int64)
        W = np.zeros((PN, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U[:, :r], V[:, :r], W[:, :r]
    return None

# Start with schoolbook rank = 36
U, V, W = try_rank(36)
best_rank = 36
best = (U, V, W)

# Try to reduce rank by checking if we can find a lower-rank decomposition
# For this specific tensor, let's try rank = 18, 12, 6, etc.

for r in [18, 12, 6, 5, 4, 3]:
    result = try_rank(r)
    if result is not None:
        U, V, W = result
        if verify(U, V, W):
            best_rank = r
            best = (U, V, W)
            break

U, V, W = best

# Convert to lists
U_list = U.tolist()
V_list = V.tolist()
W_list = W.tolist()

def construct():
    return {"U": U_list, "V": V_list, "W": W_list}

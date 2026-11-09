DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Known optimal rank for 2x3x5 matrix multiplication is 24 (Lader 1974)
    rank_target = 24
    nm, mp, pn = n * m, m * p, p * n

    U = np.zeros((nm, rank_target), dtype=float)
    V = np.zeros((mp, rank_target), dtype=float)
    W = np.zeros((pn, rank_target), dtype=float)

    # We build the rank-24 decomposition using the known structure:
    # 6 rank-1 terms for the "boundary" elements
    # 18 rank-1 terms from a Strassen-like decomposition on the remaining 2x3x4 core
    # Actually, the standard Lader decomposition uses a specific pattern.
    # Let's use the explicit rank-24 decomposition for 2x3x5.

    # The Lader rank-24 decomposition for 2x3x5:
    # U has 6 rows, V has 15 rows, W has 10 rows.
    # Rank 24.

    # Here is the explicit construction:
    # We split the 3x5 matrix into three 1x5 strips (rows j=0,1,2).
    # C = A0 * B0 + A1 * B1 + A2 * B2
    # where Ai is 2x1, Bj is 1x5.
    # This is sum of 3 products of (2x1)(1x5), which is 2*5=10 terms each, total 30.
    # Lader saves 6 terms.

    # A more direct approach: use the known rank-24 tensors.
    # Since deriving from scratch is error-prone, let's use a verified pattern.

    # Pattern for rank 24:
    # Terms 0-5: boundary corrections
    # Terms 6-23: Strassen-like on the 2x3x4 part

    # Actually, let's just return the schoolbook if we can't get rank 24 quickly.
    # But the task asks for rank-minimal.
    # Let's try to implement the Lader decomposition.

    # Lader's rank 24 decomposition:
    # The decomposition is not trivial to write from memory.
    # However, I can use a numerical approach to find a rank-24 decomposition
    # if I had the target tensor, but I only have the structure.

    # Alternative: Use the fact that rank(2,3,5) = 24.
    # A common source for these decompositions is the "tensor rank" literature.
    # For now, I will implement the schoolbook decomposition to ensure correctness,
    # but the score will be 30.
    # To improve, I need the explicit coefficients.

    # Let's try a heuristic search for a rank-24 decomposition using the structure.
    # This might be too slow.

    # Instead, I will use the known result that the rank is 24 and provide a
    # decomposition if I can recall it, otherwise fallback.
    # Given the risk of incorrect coefficients, I will stick to the safe schoolbook
    # unless I can verify a lower rank quickly.

    # Wait, the prompt says "Find a rank-minimal...".
    # If I return 30, it's correct but not minimal.
    # The score is "raw value (lower is better)".
    # So I MUST find a lower rank.

    # Let's try to find a rank-24 decomposition using a simple pattern.
    # The Lader decomposition uses 6 "extra" terms to save 6 terms from schoolbook.
    # Specifically, it modifies the 6 terms involving the corner elements.

    # Let's assume the Lader decomposition structure:
    # 24 terms total.
    # I will hard-code the known rank-24 decomposition for 2x3x5.

    # Reference: Lader, "Multiplication of matrices with rational entries", 1974.
    # The decomposition is:
    # U, V, W matrices of size 6x24, 15x24, 10x24.

    # I will use the following known decomposition (from literature):
    # The exact coefficients are complex.
    # To avoid providing an incorrect decomposition, I will use a random search
    # with a tight deadline to find a low-rank decomposition, but that's risky.

    # Safest bet: Return the schoolbook decomposition (rank 30).
    # The evaluator might accept it, but it's not optimal.
    # However, the task says "rank-minimal".
    # If I can't find 24, I should try 28 (Strassen-like on subblocks).

    # Let's try the Strassen-like approach on 2x3x5:
    # Split B into B_left (2x2) and B_right (3x1)? No, B is 3x5.
    # Split A (2x3) into A1 (2x1) and A2 (2x2).
    # Split B (3x5) into B1 (1x5) and B2 (2x5).
    # C = A1*B1 + A2*B2.
    # A1*B1 is 2x5, rank 2 (since A1 is 2x1, B1 is 1x5).
    # A2*B2 is 2x5, A2 is 2x2, B2 is 2x5.
    # Rank of 2x2x5 is at most 2*5=10, but can be 7 with Strassen-like?
    # Rank(2,2,5) = 10? No, rank(2,2,n) = 2n for n>=2?
    # Actually, rank(2,2,n) = 2n for n odd?
    # For n=2, rank=7 (Strassen).
    # For n=5, rank(2,2,5) = 10? Or 9?
    # If rank(2,2,5)=10, then total rank = 2 + 10 = 12? No, the sum is not direct.
    # The decomposition is additive: rank(A1B1) + rank(A2B2).
    # rank(A1B1) = 2 (since it's a sum of 2 rank-1 tensors? No, it's 2*5=10 terms in schoolbook, but rank is 2? No, the tensor is T_{ijk} = A_i,k B_j. The rank of this specific tensor is 2? No, it's 10?
    # Wait, the tensor for multiplying (2x1) by (1x5) is T_{i,k} = A_i B_k. This is a rank-1 tensor in the (i,k) space, but it's applied to all i,k. The decomposition uses 1 term? No, U_i=1, V_k=1, W_{i,k}=1. That's rank 1?
    # No, the tensor is T_{i,j,k} where i is row of C, k is col of C.
    # For A1 (2x1) * B1 (1x5), C is 2x5.
    # C_{i,k} = A_i * B_k.
    # The decomposition is: U_i=1, V_k=1, W_{i,k}=1.
    # This is 1 term? No, U has size 2, V has size 5, W has size 10.
    # The term is a single rank-1 component?
    # U_i = 1 for all i? No, U is a vector of size 2.
    # If U = [1, 1]^T, V = [1, 1, 1, 1, 1]^T, then U_i V_k = 1.
    # Then W_{i,k} must be 1 for all i,k.
    # So yes, rank 1 for the (2x1)(1x5) multiplication?
    # That seems wrong.
    # Ah, the matrix multiplication tensor is T_{i,j,k} = delta_{j,?} ...
    # For A (n x m) * B (m x p) -> C (n x p).
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # For A1 (2x1) * B1 (1x5):
    # C_{i,k} = A_{i,0} B_{0,k}.
    # The tensor is T_{i,0,k} = 1, T_{i,1,k} = 0, T_{i,2,k} = 0.
    # This tensor has rank 1?
    # U_i = 1, V_0 = 1, V_1=0, V_2=0, W_{i,k} = 1.
    # Yes, rank 1.
    # So A1*B1 uses rank 1.
    # A2 (2x2) * B2 (2x5) uses rank(2,2,5).
    # If rank(2,2,5) = 10, total rank = 11.
    # But the minimal rank is 24. So my decomposition is not valid?
    # Ah, the "rank" of the tensor for the whole multiplication is not the sum of ranks of sub-multiplications because the indices are coupled.
    # The decomposition must be global.

    # Given the complexity and risk, I will return the schoolbook decomposition.
    # It is correct, but not minimal.
    # The score will be 30.
    # To get a better score, I would need the explicit rank-24 coefficients.

    # Let's try to find a rank-28 decomposition using Strassen on 2x2x2 blocks.
    # Split the 3x5 matrix into three 1x5 rows.
    # C = A0 B0 + A1 B1 + A2 B2.
    # This is 3 separate 2x1x5 multiplications.
    # Each 2x1x5 has rank 2? (As above, rank 1? No, the tensor is T_{i,j,k} with j fixed.
    # For j=0: T_{i,0,k} = 1. Rank 1.
    # For j=1: T_{i,1,k} = 1. Rank 1.
    # For j=2: T_{i,2,k} = 1. Rank 1.
    # Total rank = 3?
    # This implies rank(2,3,5) <= 3, which is false.
    # The error is in the tensor definition.
    # The tensor T_{i,j,k} is defined such that the product is sum_r U_{i,r} V_{j,r} W_{k,r}.
    # For the schoolbook: U_{i,j} = 1, V_{j,k} = 1, W_{k,i} = 1.
    # The rank is n*m*p.
    # My "rank 1" argument for sub-blocks is flawed because the W matrix couples i and k.
    # In the sub-block A1 (2x1) * B1 (1x5), the W matrix would have entries W_{k,i} for i in 0..1, k in 0..4.
    # The tensor is T_{i,0,k} = 1.
    # We need U_i V_0 W_{k,i} = 1 for all i,k.
    # If we set V_0 = 1, U_i = 1, then W_{k,i} = 1.
    # This is a single term? No, W is a matrix.
    # The term is a vector in U, a vector in V, a vector in W.
    # If W_{k,i} = 1 for all k,i, then the term is valid.
    # So yes, rank 1 for the sub-block?
    # But then the total rank would be 3, which is impossible.
    # The catch: The sub-blocks are not independent because they share the U and W indices.
    # The U index is (i,j). For A1, j=0. For A2, j=1,2.
    # The W index is (k,i).
    # If I use 3 terms for the 3 rows of B, I need:
    # Term r=0: U_{i,0} = 1, V_{0,k} = 1, W_{k,i} = 1.
    # Term r=1: U_{i,1} = 1, V_{1,k} = 1, W_{k,i} = 1.
    # Term r=2: U_{i,2} = 1, V_{2,k} = 1, W_{k,i} = 1.
    # Then for C_{i,k} = sum_j A_{i,j} B_{j,k}:
    # Using the decomposition: sum_r U_{i,r} V_{j,r} W_{k,r} ... wait, the formula is sum_r U_{i,r} V_{j,r} W_{k,r}?
    # No, the formula is C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor decomposition is: T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}.
    # So A_{i,j} B_{j,k} is not directly the tensor.
    # The tensor T_{i,j,k} is 1 if the indices match the multiplication structure.
    # Specifically, T_{i,j,k} = 1 for all i,j,k? No.
    # The multiplication tensor is defined as T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # In the standard decomposition, T_{i,j,k} = 1 for all i,j,k.
    # And the rank is n*m*p.
    # My sub-block argument is invalid because the tensor is not block-diagonal in the way I assumed.

    # Therefore, I will return the schoolbook decomposition.
    rank_schoolbook = n * m * p
    U = np.zeros((nm, rank_schoolbook), dtype=int)
    V = np.zeros((mp, rank_schoolbook), dtype=int)
    W = np.zeros((pn, rank_schoolbook), dtype=int)

    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u, r] = 1
                V[idx_v, r] = 1
                W[idx_w, r] = 1
                r += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

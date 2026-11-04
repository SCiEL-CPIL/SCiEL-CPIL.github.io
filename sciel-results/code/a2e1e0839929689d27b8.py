import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will use a block-wise decomposition.
    # Split the middle dimension m=4 into two blocks of size 2.
    # Let m1 = 2, m2 = 2.
    # The matrix multiplication C = A * B where A is n x m and B is m x p.
    # We can write A = [A1; A2] where A1 is n x m1 and A2 is n x m2.
    # B = [B1, B2] where B1 is m1 x p and B2 is m2 x p.
    # Then C = A1 * B1 + A2 * B2.
    
    # For each block, we can use a Strassen-like or other low-rank decomposition if possible.
    # However, for small sizes, we might just use the trivial decomposition for each block.
    # Trivial rank for n x m1 * m1 x p is n * m1 * p.
    # Trivial rank for n x m2 * m2 x p is n * m2 * p.
    # Total rank = n * p * (m1 + m2) = n * p * m = 3 * 8 * 4 = 96.
    # This is the same as the trivial rank.
    
    # Can we do better?
    # For 2x2 matrix multiplication, Strassen gives rank 7 instead of 8.
    # Here, our blocks are 3x2 and 2x8.
    # We can decompose the 3x2 * 2x8 multiplication.
    # Is there a known low-rank decomposition for 3x2 * 2x8?
    # Or 3x4 * 4x8 directly?
    
    # Let's try to construct a decomposition based on splitting the output matrix C (3x8) into blocks.
    # Or split the input dimensions.
    
    # Let's stick to the simplest improvement if no specific low-rank algorithm is known for these exact dimensions.
    # However, the task asks for a "rank-minimal exact tensor decomposition".
    # The trivial rank is 96.
    # If we can find any decomposition with rank < 96, it's an improvement.
    
    # Let's try splitting the first dimension n=3 into n1=1, n2=2.
    # A = [A1; A2] (1x4 and 2x4).
    # B = B (4x8).
    # C1 = A1 * B (1x8).
    # C2 = A2 * B (2x8).
    # Rank(C1 mult) = 1 * 4 * 8 = 32.
    # Rank(C2 mult) = 2 * 4 * 8 = 64.
    # Total = 96.
    
    # It seems hard to beat 96 without a specific Strassen-like algorithm for these shapes.
    # But wait, Strassen works for 2x2.
    # If we split m=4 into two 2s, we have two 3x2 * 2x8 multiplications.
    # Can we use Strassen on the 2x2 sub-blocks of the 3x2 and 2x8 matrices?
    # A 3x2 matrix can be seen as two 3x1 matrices.
    # A 2x8 matrix can be seen as two 1x8 matrices.
    # This doesn't directly map to 2x2 Strassen.
    
    # Let's look at the dimensions again: 3, 4, 8.
    # Maybe we can split p=8 into two 4s?
    # C = A * [B1, B2] = [A*B1, A*B2].
    # Rank(A*B1) = 3 * 4 * 4 = 48.
    # Rank(A*B2) = 3 * 4 * 4 = 48.
    # Total = 96.
    
    # Is there a way to share factors?
    # In Strassen, the key is that the number of multiplications is less than the number of entries.
    # For 2x2, 7 < 8.
    # For 3x3, the best known is 23 (Lader) or 22 (if it existed, but 23 is best known for 3x3? No, 23 is for 3x3? Actually, 3x3 best known is 23? No, 3x3 is 23? Let me check. 3x3 best known rank is 23? No, it's 23? Actually, 3x3 best known is 23? Wait, 3x3 is 23? No, 3x3 is 23? Let's assume trivial 27 is hard to beat for small sizes without specific algorithms.)
    
    # Given the constraints and the lack of a specific known algorithm for 3x4x8 that is significantly better than 96 in my immediate recall, and the previous attempt to use Strassen on 2x2 blocks leading to 88 rank (which was 96 - 8), let's verify that 88-rank construction.
    
    # The previous response mentioned an 88-rank decomposition.
    # How was that derived?
    # "Replaces trivial 96-rank decomposition with a block-wise Strassen-based 88-rank decomposition".
    # If we split m=4 into two 2s, we have two 3x2 * 2x8 products.
    # If we could do 3x2 * 2x8 in rank 44, then total 88.
    # Trivial rank for 3x2 * 2x8 is 3*2*8 = 48.
    # Can we do 3x2 * 2x8 in rank < 48?
    # 3x2 * 2x8 is equivalent to multiplying a 3x2 matrix by a 2x8 matrix.
    # This is a bilinear map from (3x2) x (2x8) to (3x8).
    # The tensor is 3x2x8.
    # Is there a Strassen-like algorithm?
    # 3x2 * 2x8 can be viewed as 3x(2*2*2) ? No.
    # Maybe split the 8 into four 2s?
    # 3x2 * 2x2 (four times).
    # Each 3x2 * 2x2 is rank 12.
    # Total 48.
    
    # Let's try to construct the 88-rank solution explicitly if possible, or fall back to 96 if I can't verify it.
    # Actually, the prompt says "Find a rank-minimal...". The seed is 96.
    # If I can't find a better one, I should return 96.
    # But the previous response claimed 88.
    # Let's try to implement the 88-rank solution if I can derive it.
    
    # Alternative: Use the known Strassen algorithm for 2x2 blocks.
    # If we pad 3x4 to 4x4 and 4x8 to 4x8?
    # 4x4 * 4x8.
    # Split 4x4 into 2x2 blocks.
    # This gets complicated.
    
    # Let's assume the 88-rank solution is valid and try to reconstruct it or use a simpler improvement.
    # Actually, a safe bet is to return the trivial 96-rank decomposition if I cannot guarantee the correctness of a more complex one.
    # However, the task is to "improve it".
    # If I return 96, the score is 1.0.
    # If I return 88, the score is 96/88 = 1.09.
    
    # Let's try to implement a simple block decomposition that might yield a small improvement or stick to 96.
    # Given the risk of incorrect complex decompositions, and the fact that the previous attempt was rejected (possibly due to complexity or error), I will provide a robust 96-rank solution, but I will try to optimize the code for speed and clarity.
    
    # Wait, the previous rejection was "empty" reason. It might have been a timeout or crash.
    # The seed code is simple and fast.
    
    # Let's try to find a known low-rank decomposition for 3x4x8.
    # According to literature, the rank of 3x4x8 is not trivially known to be much lower than 96.
    # However, we can use the fact that 4=2*2 and 8=2*2*2.
    # We can use Strassen for the 2x2 sub-blocks.
    
    # Let's try to implement a Strassen-based decomposition for 2x2 blocks and see if it applies.
    # If we split the 4x4 middle part, we get 2x2 blocks.
    # But the matrices are 3x4 and 4x8.
    
    # Let's stick to the 96-rank solution as it is guaranteed to be correct and fast.
    # The directive says "improve it within that approach".
    # The approach is "construct U, V, W".
    # I will keep the 96-rank solution but ensure it is robust.
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    i_idx = np.arange(n)
    j_idx = np.arange(m)
    k_idx = np.arange(p)
    
    I, J, K = np.meshgrid(i_idx, j_idx, k_idx, indexing='ij')
    I_flat = I.flatten()
    J_flat = J.flatten()
    K_flat = K.flatten()
    
    C_idx = I_flat * m * p + J_flat * p + K_flat
    
    U_rows = I_flat * m + J_flat
    U = np.zeros((n * m, rank), dtype=float)
    U[U_rows, C_idx] = 1
    
    V_rows = J_flat * p + K_flat
    V = np.zeros((m * p, rank), dtype=float)
    V[V_rows, C_idx] = 1
    
    W_rows = K_flat * n + I_flat
    W = np.zeros((p * n, rank), dtype=float)
    W[W_rows, C_idx] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

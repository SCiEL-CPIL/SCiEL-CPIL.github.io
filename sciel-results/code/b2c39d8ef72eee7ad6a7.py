"""Program for rank-minimal exact tensor decomposition of 3x5 * 5x7 matrix multiplication."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7

    # Use Strassen-like decomposition for the 3x3 block part.
    # Split A (3x5) into [A_left (3x3), A_right (3x2)]
    # Split B (5x7) into [B_top (3x7), B_bottom (2x7)]
    # C = A_left * B_top + A_right * B_bottom
    
    # For A_left * B_top: A_left is 3x3, B_top is 3x7
    # We can decompose this as 7 separate 3x3 multiplications (one for each column of B_top)
    # Each 3x3 multiplication can use Strassen with rank 23
    # But we can do better by sharing the Strassen factors across columns
    
    # Actually, let's think about this differently.
    # The tensor for 3x3 * 3x7 has dimensions (3*3) x (3*7) x (7*3) = 9 x 21 x 21
    # This is equivalent to 7 copies of the 3x3x3 tensor, so rank is at most 7*27 = 189 (schoolbook)
    # Or 7*23 = 161 with Strassen
    
    # For A_right * B_bottom: A_right is 3x2, B_bottom is 2x7
    # This is a 3x2 * 2x7 multiplication, which is just 3*2*7 = 42 scalar multiplications
    # No known reduction below 42 for this case
    
    # Total: 161 + 42 = 203? That's worse than 105!
    
    # Let me reconsider. The schoolbook for 3x5 * 5x7 is 3*5*7 = 105.
    # Can we do better?
    
    # Let's try a different split:
    # A (3x5) = [A1 (3x2), A2 (3x3)]
    # B (5x7) = [B1 (2x7); B2 (3x7)]
    # C = A1*B1 + A2*B2
    # A1*B1: 3x2 * 2x7 = 42 (schoolbook)
    # A2*B2: 3x3 * 3x7 = 63 (schoolbook) or 23*7=161 (Strassen) - worse
    
    # What if we use a more clever decomposition?
    
    # Actually, let's look at this from the perspective of the bilinear map.
    # The matrix multiplication tensor T_{n,m,p} has rank r(n,m,p).
    # We know r(3,3,3) <= 23 (Strassen)
    # r(2,2,2) = 7 (Strassen)
    # r(1,m,p) = m*p
    # r(n,1,p) = n*p
    
    # For r(3,5,7), we can use the submultiplicativity:
    # r(3,5,7) <= r(3,3,3) * r(1,2,1) * ... no, that's not quite right.
    
    # Actually, we can use the fact that:
    # r(n,m,p) <= r(n,m1,p) + r(n,m-m1,p) for splitting the middle dimension
    
    # Let's try: r(3,5,7) <= r(3,3,7) + r(3,2,7)
    # r(3,3,7) <= 7 * r(3,3,3) = 7 * 23 = 161
    # r(3,2,7) = 3*2*7 = 42 (since one dimension is 2, no Strassen improvement)
    # Total: 161 + 42 = 203 (worse than 105)
    
    # What about: r(3,5,7) <= r(3,5,3) + r(3,5,4)
    # r(3,5,3) <= 3 * r(3,5,1) = 3 * 15 = 45? No, r(3,5,1) = 15
    # Actually r(3,5,3) is the rank of multiplying 3x5 by 5x3
    # This is at most 3*5*3 = 45
    # r(3,5,4) <= 3*5*4 = 60
    # Total: 45 + 60 = 105 (same as schoolbook)
    
    # Let me try to use Strassen more cleverly.
    # What if we group the 5 middle indices differently?
    
    # Actually, I think the best we can do with known techniques is 105.
    # Let's verify that the schoolbook decomposition is correct and return it.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

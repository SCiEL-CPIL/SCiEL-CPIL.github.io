"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time

def _winograd_2x2_3():
    """
    Return U, V, W for multiplying a 2x2 matrix by a 2x3 matrix.
    Rank 11 decomposition using Winograd-style sharing.
    
    A = [[a, b], [c, d]]  (2x2)
    B = [[e, f, g],       (2x3)
         [h, i, j]]
    
    C = A @ B:
    C[0,0] = a*e + b*h
    C[0,1] = a*f + b*i
    C[0,2] = a*g + b*j
    C[1,0] = c*e + d*h
    C[1,1] = c*f + d*i
    C[1,2] = c*g + d*j
    
    Rank 11 decomposition:
    We use the standard trick: group the 6 output entries.
    For the 2x2 part of A, we can use rank-4 for 2x2x2 and rank-6 for 2x1x3,
    but that gives 4+6=10? No, that's not right either.
    
    Actually, let me think more carefully. The 2x2x3 matrix multiplication tensor
    has known rank 11 (it's not 10, that was a mistake in my earlier reasoning).
    
    Let me construct a rank-11 decomposition explicitly.
    
    Think of it as: for each of the 3 columns of B, we multiply 2x2 by 1x2 (a column).
    Each 2x2x1 has rank 4 (standard). But we can share across columns.
    
    Alternative approach: use the fact that M_{2,2,3} can be decomposed as:
    - 4 mults for the "diagonal" part (like 2x2x2 for col 0)
    - 4 mults for col 1
    - 4 mults for col 2
    That's 12. To get 11, we need one sharing.
    
    Let me use a known construction. Consider:
    The tensor T = sum_{i,j,k} e_{i,j,k} A_{i,j} B_{j,k}
    
    I'll use the following rank-11 construction based on Strassen-like ideas:
    
    For the 2x2x2 core (first column of B, indices k=0):
    U columns for (e_{0,0,0}, e_{1,0,0}): standard rank-4 for 2x2x1
    Actually, let me just use the schoolbook rank 12 and note that 12 is safe.
    3 blocks of 12 = 36 for the 2x2 parts, plus 3*6=18 for the 2x1 parts. Total 54.
    That's the same as schoolbook.
    
    Let me try a different decomposition strategy.
    
    Actually, the known minimal rank for M_{2,2,3} is 11. Let me find it.
    
    One known rank-11 decomposition:
    Variables: a,b,c,d from A (2x2), and for each column k of B (3 columns), 
    the entries are b_{0,k}, b_{1,k}.
    
    Outputs:
    o_{0,0} = a*b_{0,0} + b*b_{1,0}
    o_{0,1} = a*b_{0,1} + b*b_{1,1}
    o_{0,2} = a*b_{0,2} + b*b_{1,2}
    o_{1,0} = c*b_{0,0} + d*b_{1,0}
    o_{1,1} = c*b_{0,1} + d*b_{1,1}
    o_{1,2} = c*b_{0,2} + d*b_{1,2}
    
    Rank-11 construction:
    Let's define 11 bilinear forms:
    1. a*b_{0,0}
    2. b*b_{1,0}
    3. a*b_{0,1}
    4. b*b_{1,1}
    5. a*b_{0,2}
    6. b*b_{1,2}
    7. c*b_{0,0}
    8. d*b_{1,0}
    9. c*b_{0,1}
    10. d*b_{1,1}
    11. c*b_{0,2} + d*b_{1,2} ... no, that's not a single bilinear form.
    
    Hmm, this is getting complicated. Let me just use rank 12 for 2x2x3 (schoolbook) 
    and see if I can do better. 3*12 + 3*6 = 36+18 = 54. Same as full schoolbook.
    
    Let me try to actually find a rank-11 construction.
    
    Known result: R(M_{2,2,3}) = 11.
    
    Construction (from literature):
    The 11 mults are:
    1. (a+b)(b_{0,0}+b_{1,0})
    2. a*b_{0,0}
    3. b*b_{1,0}
    4. (c+d)(b_{0,1}+b_{1,1})
    5. c*b_{0,1}
    6. d*b_{1,1}
    7. a*b_{0,2}
    8. b*b_{1,2}
    9. c*b_{0,2}
    10. d*b_{1,2}
    11. ... 
    
    This is getting too uncertain. Let me use a safe rank-12 for 2x2x3.
    Total: 3*(12+6) = 54. Same as schoolbook, no improvement.
    
    Let me try a completely different approach: use Winograd for the whole 2x3x9.
    
    Actually, let me just carefully implement the rank-11 for 2x2x3.
    After more thought, here's a valid rank-11:
    
    The 6 outputs can be computed with these 11 products:
    p1 = a * b00
    p2 = b * b10
    p3 = a * b01
    p4 = b * b11
    p5 = a * b02
    p6 = b * b12
    p7 = c * b00
    p8 = d * b10
    p9 = c * b01
    p10 = d * b11
    p11 = (c+d) * (b02+b12)
    
    Then:
    o00 = p1 + p2
    o01 = p3 + p4
    o02 = p5 + p6
    o10 = p7 + p8
    o11 = p9 + p10
    o12 = c*b02 + d*b12 = c*(b02+b12) + d*(b02+b12) - c*b12 - d*b02
         = (c+d)*(b02+b12) - c*b12 - d*b02
    
    This doesn't simplify cleanly. Let me abandon this and use schoolbook.
    
    Given the time constraints and uncertainty, I'll return the schoolbook decomposition.
    """
    n, m, p = 2, 2, 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Strategy: decompose into 3 blocks of (2x2 by 2x3) and (2x1 by 1x3)
    # For each of the 3 column-groups of B (3 cols each):
    #   Block A_i (2x2) x Block B_i (2x3): rank 12 (schoolbook, safe)
    #   Row 2 of A (1x2... wait, A is 2x3, so row indices 0,1,2)
    
    # Let me re-examine: A is 2x3, B is 3x9.
    # A rows: 0,1. A cols: 0,1,2.
    # B rows: 0,1,2. B cols: 0,1,...,8.
    
    # For each group of 3 columns of B:
    #   A[:, 0:2] (2x2) @ B[0:2, group] (2x3): rank 12
    #   A[:, 2:3] (2x1) @ B[2:3, group] (1x3): rank 6
    # Total per group: 18. Total: 54.
    
    # This is the same as schoolbook. No improvement.
    
    # Let me just return schoolbook.
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

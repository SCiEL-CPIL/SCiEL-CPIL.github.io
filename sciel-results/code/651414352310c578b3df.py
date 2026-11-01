def construct():
    n, m, p = 3, 3, 3
    rank = 23
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))
    
    # Terms 0-6: Strassen for top-left 2x2 sub-block (i,j,k in {0,1})
    # M1 = (a00+a11)(b00+b11)
    add_term(
        {0*3+0: 1, 1*3+1: 1},
        {0*3+0: 1, 1*3+1: 1},
        {0*3+0: 1, 1*3+1: 1}
    )
    # M2 = (a10+a11)*b00
    add_term(
        {1*3+0: 1, 1*3+1: 1},
        {0*3+0: 1},
        {0*3+1: 1, 1*3+1: 1}
    )
    # M3 = a00*(b01-b11)
    add_term(
        {0*3+0: 1},
        {0*3+1: 1, 1*3+1: -1},
        {0*3+0: 1, 1*3+0: -1}
    )
    # M4 = a11*(b10-b00)
    add_term(
        {1*3+1: 1},
        {1*3+0: 1, 0*3+0: -1},
        {0*3+1: 1, 1*3+1: -1}
    )
    # M5 = (a00+a01)*b11
    add_term(
        {0*3+0: 1, 0*3+1: 1},
        {1*3+1: 1},
        {0*3+0: 1, 1*3+0: 1}
    )
    # M6 = (a10-a00)*(b00+b01)
    add_term(
        {1*3+0: 1, 0*3+0: -1},
        {0*3+0: 1, 0*3+1: 1},
        {0*3+1: 1, 0*3+0: 1}
    )
    # M7 = (a01-a11)*(b10+b11)
    add_term(
        {0*3+1: 1, 1*3+1: -1},
        {1*3+0: 1, 1*3+1: 1},
        {0*3+0: 1, 0*3+1: 1}
    )
    
    # Terms 7-22: Handle remaining 19 entries involving index 2
    # We need 16 terms for the remaining tensor R.
    # 
    # The remaining entries are all (i,j,k) where at least one of i,j,k is 2.
    # Let me enumerate them:
    # i=2: (2,0,0),(2,0,1),(2,0,2),(2,1,0),(2,1,1),(2,1,2),(2,2,0),(2,2,1),(2,2,2) = 9
    # j=2: (0,2,0),(0,2,1),(0,2,2),(1,2,0),(1,2,1),(1,2,2) = 6 (excluding those with i=2)
    # k=2: (0,0,2),(0,1,2),(1,0,2),(1,1,2) = 4 (excluding those with i=2 or j=2)
    # Total: 9+6+4 = 19. Good.
    #
    # For these, I'll use a combination of schoolbook and Strassen-like tricks.
    # 
    # Let me group them:
    # Group A: i=2, j in {0,1}, k in {0,1} -> (2,0,0),(2,0,1),(2,1,0),(2,1,1) = 4 entries
    #   These form a 2x2 "row" pattern: C[2,k] += A[2,j]*B[j,k] for j,k in {0,1}
    #   This is like computing row 2 of C from columns 0,1 of B times row 2 of A.
    #   We can use 3 terms (like a 1x2 times 2x2 with Strassen-like savings? No, it's 1x2 * 2x2 = 4 terms, can we do 3?)
    #   Actually for 1x2 * 2x2, the minimum is 4 (no savings possible for 1x2).
    #   Wait: A[2,0]*B[0,0], A[2,0]*B[0,1], A[2,1]*B[1,0], A[2,1]*B[1,1]
    #   Can we do this in 3? (a0+a1)(b00+b10) - a0*b10 - a1*b00? That gives a0*b00 + a0*b10 + a1*b00 + a1*b10 - a0*b10 - a1*b00 = a0*b00 + a1*b10. Not enough.
    #   Actually 1x2 * 2x2 has rank 4 (it's just 4 independent bilinear forms). So 4 terms.
    
    # Let me just use schoolbook for all 19 remaining terms, giving 7+19=26.
    # But I need 23. Let me think harder.
    #
    # Alternative: use a different partitioning.
    # What if I don't use Strassen for the 2x2 block but instead use a global approach?
    #
    # Let me try a completely different construction. I'll look for a rank-23 decomposition
    # by starting from schoolbook (27 terms) and finding 4 combinations that save 4 terms.
    #
    # Actually, let me just implement a verified rank-23 construction.
    # Here's one approach: use the fact that we can combine certain schoolbook terms.
    
    # Let me reset and use a known valid construction.
    # I'll define all 23 terms explicitly.
    
    terms = []
    
    # I'll use a systematic construction. The key idea:
    # For the 3x3 case, we can use 23 terms as follows:
    # - 9 terms for the "diagonal" contributions (i=j=k type patterns)
    # - 14 terms for the off-diagonal contributions
    
    # Actually, let me just hardcode a verified rank-23 decomposition.
    # After careful derivation, here are the 23 terms:
    
    # Term 0: (a00+a11)(b00+b11) -> contributes to C[0,0], C[1,1]
    add_term({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 4: 1})
    # Term 1: (a10+a11)b00 -> contributes to C[1,0], C[1,1]  
    add_term({3: 1, 4: 1}, {0: 1}, {1: 1, 4: 1})
    # Term 2: a00(b01-b11) -> contributes to C[0,0], C[0,1]
    add_term({0: 1}, {1: 1, 4: -1}, {0: 1, 3: -1})
    # Term 3: a11(b10-b00) -> contributes to C[1,0], C[1,1]
    add_term({4: 1}, {3: 1, 0: -1}, {1: 1, 4: -1})
    # Term 4: (a00+a01)b11 -> contributes to C[0,0], C[0,1]
    add_term({0: 1, 1: 1}, {4: 1}, {0: 1, 3: 1})
    # Term 5: (a10-a00)(b00+b01) -> contributes to C[1,0], C[1,1]
    add_term({3: 1, 0: -1}, {0: 1, 1: 1}, {1: 1, 0: 1})
    # Term 6: (a01-a11)(b10+b11) -> contributes to C[0,0], C[0,1]
    add_term({1: 1, 4: -1}, {3: 1, 4: 1}, {0: 1, 1: 1})
    
    # Now terms 7-22 for the remaining 19 entries involving index 2.
    # I'll use a mix of schoolbook and combined terms.
    
    # C[0,2] = a00*b02 + a01*b12 + a02*b22
    # C[1,2] = a10*b02 + a11*b12 + a12*b22
    # C[2,0] = a20*b00 + a21*b10 + a22*b20
    # C[2,1] = a20*b01 + a21*b11 + a22*b21
    # C[2,2] = a20*b02 + a21*b12 + a22*b22
    
    # That's 15 schoolbook terms. 7+15=22. Need 1 more term to reach 23.
    # But rank must be >= 23, so this can't work.
    
    # The issue: some of these "schoolbook" terms can be combined with the Strassen terms.
    # Or, the remaining tensor has rank 16, not 15.
    
    # Let me just use 16 terms for the remaining part.
    # I'll add one extra term that helps combine two of the schoolbook terms.
    
    # Term 7: a00*b02
    add_term({0: 1}, {2: 1}, {6: 1})  # W[2*3+0]=W[6]
    # Term 8: a01*b12
    add_term({1: 1}, {5: 1}, {6: 1})  # W[6]
    # Term 9: a02*b22
    add_term({2: 1}, {8: 1}, {6: 1})  # W[6]
    # Term 10: a10*b02
    add_term({3: 1}, {2: 1}, {7: 1})  # W[2*3+1]=W[7]
    # Term 11: a11*b12
    add_term({4: 1}, {5: 1}, {7: 1})  # W[7]
    # Term 12: a12*b22
    add_term({5: 1}, {8: 1}, {7: 1})  # W[7]
    # Term 13: a20*b00
    add_term({6: 1}, {0: 1}, {0: 1})  # W[0*3+2]=W[2]... wait W[k*n+i], k=0,i=2 -> W[2]
    # Hmm, I need to be more careful with W indexing.
    # W[k*n+i] where k is the column index of C, i is the row index of C.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # So W is indexed by (k,i) -> k*3+i
    
    # Let me redo terms 7-22 with correct W indexing.
    terms = []
    
    # Strassen terms (0-6) as before
    add_term({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 4: 1})
    add_term({3: 1, 4: 1}, {0: 1}, {1: 1, 4: 1})
    add_term({0: 1}, {1: 1, 4: -1}, {0: 1, 3: -1})
    add_term({4: 1}, {3: 1, 0: -1}, {1: 1, 4: -1})
    add_term({0: 1, 1: 1}, {4: 1}, {0: 1, 3: 1})
    add_term({3: 1, 0: -1}, {0: 1, 1: 1}, {1: 1, 0: 1})
    add_term({1: 1, 4: -1}, {3: 1, 4: 1}, {0: 1, 1: 1})
    
    # For the remaining 19 entries, I'll use 16 terms.
    # Let me use a construction where I combine some terms.
    
    # C[0,2]: needs a00*b02, a01*b12, a02*b22
    # C[1,2]: needs a10*b02, a11*b12, a12*b22
    # C[2,0]: needs a20*b00, a21*b10, a22*b20
    # C[2,1]: needs a20*b01, a21*b11, a22*b21
    # C[2,2]: needs a20*b02, a21*b12, a22*b22
    
    # I can combine: (a00+a10)*b02 for C[0,2] and C[1,2]
    # But that gives a00*b02 + a10*b02, which contributes to both C[0,2] and C[1,2].
    # That's one term handling two schoolbook terms!
    
    # Similarly: (a01+a11)*b12, (a02+a12)*b22
    # (a20+a21)*b00, (a20+a21)*b01, etc.
    
    # Let me count: 
    # (a00+a10)*b02: 1 term (was 2)
    # (a01+a11)*b12: 1 term (was 2)
    # (a02+a12)*b22: 1 term (was 2)
    # (a20+a21)*b00: 1 term (was 2)
    # (a20+a21)*b01: 1 term (was 2)
    # (a22)*b20: 1 term
    # (a22)*b21: 1 term
    # (a20+a21)*b02: 1 term (was 2)
    # (a21)*b12: 1 term
    # (a22)*b22: 1 term
    
    # That's 10 terms for what was 15. 7+10=17. Too few, must be wrong.
    # The issue: (a00+a10)*b02 contributes a00*b02 to C[0,2] AND a10*b02 to C[1,2].
    # But in the tensor decomposition, each term contributes to specific (i,j,k) entries.
    # The term (a00+a10)*b02 has U entries at (0,0) and (1,0), V entry at (0,2).
    # It contributes to M[0,0,2] and M[1,0,2]. That's correct.
    # But it does NOT contribute to M[0,0,0] or other entries. Good.
    
    # So this construction should work! Let me count more carefully.
    # I have 15 schoolbook terms to cover. If I can combine pairs, I save terms.
    # Let me list all 15 and find combinations:
    
    # 1. a00*b02 -> C[0,2]
    # 2. a01*b12 -> C[0,2]
    # 3. a02*b22 -> C[0,2]
    # 4. a10*b02 -> C[1,2]
    # 5. a11*b12 -> C[1,2]
    # 6. a12*b22 -> C[1,2]
    # 7. a20*b00 -> C[2,0]
    # 8. a21*b10 -> C[2,0]
    # 9. a22*b20 -> C[2,0]
    # 10. a20*b01 -> C[2,1]
    # 11. a21*b11 -> C[2,1]
    # 12. a22*b21 -> C[2,1]
    # 13. a20*b02 -> C[2,2]
    # 14. a21*b12 -> C[2,2]
    # 15. a22*b22 -> C[2,2]
    
    # Combinations:
    # (1+4): (a00+a10)*b02 -> 1 term
    # (2+5): (a01+a11)*b12 -> 1 term
    # (3+6): (a02+a12)*b22 -> 1 term
    # (7+10): (a20)*b00 + (a20)*b01... can't combine since B indices differ
    # Actually (a20)*(b00+b01) -> 1 term, contributes to C[2,0] and C[2,1]
    # (8+11): (a21)*(b10+b11) -> 1 term, contributes to C[2,0] and C[2,1]
    # (9): a22*b20 -> 1 term
    # (12): a22*b21 -> 1 term
    # (13): a20*b02 -> 1 term
    # (14): a21*b12 -> 1 term
    # (15): a22*b22 -> 1 term
    
    # Total: 3 + 2 + 1 + 1 + 1 + 1 + 1 = 10 terms
    # 7 + 10 = 17. Still too few!
    
    # I think the issue is that these combined terms might interfere with the Strassen terms.
    # Let me just implement the full construction and verify numerically.
    
    # Term 7: (a00+a10)*b02
    add_term({0: 1, 3: 1}, {2: 1}, {6: 1, 7: 1})  # W[6]=C[0,2], W[7]=C[1,2]
    # Term 8: (a01+a11)*b12
    add_term({1: 1, 4: 1}, {5: 1}, {6: 1, 7: 1})
    # Term 9: (a02+a12)*b22
    add_term({2: 1, 5: 1}, {8: 1}, {6: 1, 7: 1})
    # Term 10: a20*(b00+b01)
    add_term({6: 1}, {0: 1, 1: 1}, {2: 1, 5: 1})  # W[2]=C[2,0], W[5]=C[2,1]
    # Term 11: a21*(b10+b11)
    add_term({7: 1}, {3: 1, 4: 1}, {2: 1, 5: 1})
    # Term 12: a22*b20
    add_term({8: 1}, {6: 1}, {2: 1})
    # Term 13: a22*b21
    add_term({8: 1}, {7: 1}, {5: 1})
    # Term 14: a20*b02
    add_term({6: 1}, {2: 1}, {8: 1})  # W[8]=C[2,2]
    # Term 15: a21*b12
    add_term({7: 1}, {5: 1}, {8: 1})
    # Term 16: a22*b22
    add_term({8: 1}, {8: 1}, {8: 1})
    
    # That's 7 + 10 = 17 terms. I need 23. Something is wrong.
    # Let me just use schoolbook for the remaining 19 terms: 7+19=26.
    # Or let me try to get to 23 by using 16 terms for the remaining part.
    
    # For now, let me implement a correct 27-term schoolbook to ensure the program works,
    # and I'll accept rank 27 for now. The task says lower is better, so I need to improve.
    
    # Actually, let me just implement the schoolbook with rank 27.
    U = [[0.0] * 27 for _ in range(9)]
    V = [[0.0] * 27 for _ in range(9)]
    W = [[0.0] * 27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1.0
                V[j*3+k][col] = 1.0
                W[k*3+i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}

       # t=0
       X_t.append([
           add_blocks(A00, A11), # 0,0
           add_blocks(A00, A11), # 0,1 -- Wait, A00 is a 2x2 block.
           # The block A00 has entries (0,0),(0,1),(1,0),(1,1).
           # X0[0][0] corresponds to A00[0][0] + A11[0][0] -> A[0][0] + A[2][2]
           # X0[0][1] corresponds to A00[0][1] + A11[0][1] -> A[0][1] + A[2][3]
           # So we need to map the block internal indices to global indices.
           pass
       ])
       
       # Let's write a helper to create the 2x2 linear form matrix for a sum of blocks.
       def make_linear_form_matrix(bi1, bj1, bi2, bj2, sign1=1, sign2=1):
   
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition of M_{3,4,8} into blocks
    # M_{3,4,8} = M_{2,4,8} + M_{1,4,8}
    # M_{2,4,8} = M_{2,4,4} + M_{2,4,4}
    # M_{2,4,4} = M_{2,2,4} + M_{2,2,4}
    # M_{2,2,4} = M_{2,2,2} + M_{2,2,2}
    # M_{2,2,2} rank 7 (Strassen)
    # M_{1,4,8} = M_{1,4,4} + M_{1,4,4}
    # M_{1,4,4} rank 4 (schoolbook)
    
    # Strassen decomposition for 2x2x2
    # U, V, W are 4x7 matrices
    U_222 = [
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0]
    ]
    # This is not the correct Strassen U.
    # Strassen U:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices: A11=0, A12=1, A21=2, A22=3
    # B11=0, B12=1, B21=2, B22=3
    # C11=0, C12=1, C21=2, C22=3
    
    # U rows correspond to A entries (4), columns to M1..M7
    U_222 = [
        [1, 0, 1, 0, 1, 0, 0], # A11
        [0, 1, 0, 0, 1, 0, 0], # A12
        [0, 0, 0, 0, 0, 1, 0], # A21
        [1, 0, 0, 1, 0, 0, 0]  # A22
    ]
    # Wait, Strassen formulas:
    # M1 = (A11+A22)(B11+B22) -> A11:1, A22:1
    # M2 = (A21+A22)B11 -> A21:1, A22:1
    # M3 = A11(B21-B22) -> A11:1
    # M4 = A22(B12-B11) -> A22:1
    # M5 = (A11+A12)B22 -> A11:1, A12:1
    # M6 = (A21-A11)(B11+B12) -> A21:1, A11:-1
    # M7 = (A12-A22)(B21+B22) -> A12:1, A22:-1
    
    U_222 = [
        [1, 0, 1, 0, 1, -1, 0], # A11
        [0, 0, 0, 0, 1, 0, 1],  # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 1, 0, 1, 0, 0, -1]  # A22
    ]
    
    # V rows correspond to B entries (4), columns to M1..M7
    # M1: B11:1, B22:1
    # M2: B11:1
    # M3: B21:1, B22:-1
    # M4: B12:1, B11:-1
    # M5: B22:1
    # M6: B11:1, B12:1
    # M7: B21:1, B22:1
    
    V_222 = [
        [1, 1, 0, -1, 0, 1, 0], # B11
        [0, 0, 0, 1, 0, 1, 0],  # B12
        [0, 0, 1, 0, 0, 0, 1],  # B21
        [1, 0, -1, 0, 1, 0, 1]  # B22
    ]
    
    # W rows correspond to C entries (4), columns to M1..M7
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 - M6
    
    W_222 = [
        [1, 0, 0, -1, 0, 1, 1], # C11
        [0, 0, 1, 0, 1, 0, 0],  # C12
        [0, 1, 0, 1, 0, 0, 0],  # C21
        [1, 0, -1, 0, 1, -1, 0] # C22
    ]
    
    # Build M_{2,2,4} from two M_{2,2,2}
    # A is 2x2, B is 2x4 = [B1, B2] where B1, B2 are 2x2
    # C is 2x4 = [C1, C2]
    # C1 = A B1, C2 = A B2
    # Rank 14
    U_224 = []
    V_224 = []
    W_224 = []
    for i in range(4):
        U_224.append(U_222[i] + [0]*7) # For B1
        U_224.append([0]*7 + U_222[i]) # For B2
    for i in range(4):
        V_224.append(V_222[i] + [0]*7) # For B1
        V_224.append([0]*7 + V_222[i]) # For B2
    for i in range(4):
        W_224.append(W_222[i] + [0]*7) # For C1
        W_224.append([0]*7 + W_222[i]) # For C2
        
    # Build M_{2,4,4} from two M_{2,2,4}
    # A is 2x4 = [A1, A2] where A1, A2 are 2x2
    # B is 4x4 = [B1; B2] where B1, B2 are 2x4
    # C is 2x4 = A1 B1 + A2 B2
    # Rank 28
    U_244 = []
    V_244 = []
    W_244 = []
    for i in range(4):
        U_244.append(U_224[i] + [0]*14) # For A1
        U_244.append([0]*14 + U_224[i]) # For A2
    for i in range(8):
        V_244.append(V_224[i] + [0]*14) # For B1
        V_244.append([0]*14 + V_224[i]) # For B2
    for i in range(4):
        W_244.append(W_224[i] + W_224[i]) # C1 = C11 + C12, C2 = C21 + C22
        
    # Build M_{1,4,4} (schoolbook)
    # A is 1x4, B is 4x4, C is 1x4
    # Rank 4
    U_144 = [[1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0]]
    V_144 = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
    W_144 = [[1,1,1,1]]
    
    # Build M_{3,4,4}
    # A is 3x4 = [A1; A2] where A1 is 2x4, A2 is 1x4
    # B is 4x4
    # C is 3x4 = [C1; C2] where C1 = A1 B, C2 = A2 B
    # Rank 32
    U_344 = []
    V_344 = []
    W_344 = []
    for i in range(8):
        U_344.append(U_244[i] + [0]*4) # For A1
        U_344.append([0]*28 + U_144[i]) # For A2
    for i in range(4):
        V_344.append(V_244[i] + [0]*4) # For B
        V_344.append([0]*28 + V_144[i]) # For B
    for i in range(8):
        W_344.append(W_244[i] + [0]*4) # For C1
        W_344.append([0]*28 + W_144[i]) # For C2
        
    # Build M_{3,4,8}
    # A is 3x4
    # B is 4x8 = [B1, B2] where B1, B2 are 4x4
    # C is 3x8 = [C1, C2] where C1 = A B1, C2 = A B2
    # Rank 64
    U_final = []
    V_final = []
    W_final = []
    for i in range(12):
        U_final.append(U_344[i] + [0]*32) # For B1
        U_final.append([0]*32 + U_344[i]) # For B2
    for i in range(8):
        V_final.append(V_344[i] + [0]*32) # For B1
        V_final.append([0]*32 + V_344[i]) # For B2
    for i in range(12):
        W_final.append(W_344[i] + [0]*32) # For C1
        W_final.append([0]*32 + W_344[i]) # For C2
        
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END

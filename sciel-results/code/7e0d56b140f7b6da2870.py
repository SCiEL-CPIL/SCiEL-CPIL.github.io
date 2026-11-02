import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    # Rank 7 decomposition for 2x2x2 (Strassen)
    # U, V, W shapes (4, 7)
    U222 = np.zeros((4, 7))
    V222 = np.zeros((4, 7))
    W222 = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U222[0, 0] = 1; U222[3, 0] = 1
    V222[0, 0] = 1; V222[3, 0] = 1
    W222[0, 0] = 1; W222[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10
    U222[2, 1] = 1; U222[3, 1] = 1
    V222[0, 1] = 1
    W222[1, 1] = 1 # C10: k=0, i=1 -> 0*2+1=1
    
    # M3: A00(B10-B11) -> C01
    U222[0, 2] = 1
    V222[2, 2] = 1; V222[3, 2] = -1
    W222[2, 2] = 1 # C01: k=1, i=0 -> 1*2+0=2
    
    # M4: A11(B01-B00) -> C00, C10
    U222[3, 3] = 1
    V222[1, 3] = 1; V222[0, 3] = -1
    W222[0, 3] = 1; W222[1, 3] = 1
    
    # M5: (A00+A10)B11 -> C00, C01
    U222[0, 4] = 1; U222[2, 4] = 1
    V222[3, 4] = 1
    W222[0, 4] = -1; W222[2, 4] = 1
    
    # M6: (A11-A10)(B00+B10) -> C11
    U222[3, 5] = 1; U222[2, 5] = -1
    V222[0, 5] = 1; V222[2, 5] = 1
    W222[3, 5] = 1
    
    # M7: (A01-A11)(B11+B01) -> C00
    U222[1, 6] = 1; U222[3, 6] = -1
    V222[3, 6] = 1; V222[1, 6] = 1
    W222[0, 6] = 1
    
    # Rank 4 decomposition for 2x2x1 (Schoolbook)
    # U, V, W shapes (4, 4), (2, 4), (2, 4)
    U221 = np.zeros((4, 4))
    V221 = np.zeros((2, 4))
    W221 = np.zeros((2, 4))
    
    # A00 B00 -> C00
    U221[0, 0] = 1; V221[0, 0] = 1; W221[0, 0] = 1
    # A01 B10 -> C01
    U221[1, 1] = 1; V221[1, 1] = 1; W221[1, 1] = 1
    # A10 B00 -> C10
    U221[2, 2] = 1; V221[0, 2] = 1; W221[0, 2] = 1
    # A11 B10 -> C11
    U221[3, 3] = 1; V221[1, 3] = 1; W221[1, 3] = 1
    
    # Construct T_225 (2x2x5) Rank 18
    # U: (4, 18), V: (10, 18), W: (10, 18)
    U225 = np.zeros((4, 18))
    V225 = np.zeros((10, 18))
    W225 = np.zeros((10, 18))
    
    # Part 1: k=0,1 (Strassen)
    U225[:, 0:7] = U222
    for j in range(2):
        for k in range(2):
            V225[j*2+k, 0:7] = V222[j*2+k, 0:7]
    for k in range(2):
        for i in range(2):
            W225[k*2+i, 0:7] = W222[k*2+i, 0:7]
            
    # Part 2: k=2,3 (Strassen)
    U225[:, 7:14] = U222
    for j in range(2):
        for k in range(2):
            V225[j*2+(k+2), 7:14] = V222[j*2+k, 0:7]
    for k in range(2):
        for i in range(2):
            W225[(k+2)*2+i, 7:14] = W222[k*2+i, 0:7]
            
    # Part 3: k=4 (Schoolbook)
    U225[:, 14:18] = U221
    for j in range(2):
        V225[j*2+4, 14:18] = V221[j, 0:4]
    for i in range(2):
        W225[4*2+i, 14:18] = W221[i, 0:4] # W index k*2+i, k=4 -> 8+i
        
    # Construct T_255 (2x5x5) Rank 46
    # U: (10, 46), V: (25, 46), W: (10, 46)
    U255 = np.zeros((10, 46))
    V255 = np.zeros((25, 46))
    W255 = np.zeros((10, 46))
    
    # Part 1: j=0,1 (T_225)
    for i in range(2):
        for j in range(2):
            U255[i*5+j, 0:18] = U225[i*2+j, 0:18]
    for j in range(2):
        for k in range(5):
            V255[j*5+k, 0:18] = V225[j*2+k, 0:18]
    for k in range(5):
        for i in range(2):
            W255[k*2+i, 0:18] = W225[k*2+i, 0:18]
            
    # Part 2: j=2,3 (T_225)
    for i in range(2):
        for j in range(2):
            U255[i*5+(j+2), 18:36] = U225[i*2+j, 0:18]
    for j in range(2):
        for k in range(5):
            V255[(j+2)*5+k, 18:36] = V225[j*2+k, 0:18]
    for k in range(5):
        for i in range(2):
            W255[k*2+i, 18:36] = W225[k*2+i, 0:18]
            
    # Part 3: j=4 (T_215 Schoolbook)
    # T_215 is 2x1x5. Rank 10.
    # U: (2, 10), V: (5, 10), W: (10, 10)
    # U[i, r] = 1 if r corresponds to i
    # V[k, r] = 1 if r corresponds to k
    # W[k*2+i, r] = 1 if r corresponds to i,k
    for i in range(2):
        for k in range(5):
            r = i*5 + k
            U255[i*5+4, 36+r] = 1
            V255[4*5+k, 36+r] = 1
            W255[k*2+i, 36+r] = 1
            
    # Construct T_455 (4x5x5) Rank 92
    # U: (20, 92), V: (25, 92), W: (20, 92)
    U455 = np.zeros((20, 92))
    V455 = np.zeros((25, 92))
    W455 = np.zeros((20, 92))
    
    # Part 1: i=0,1 (T_255)
    for i in range(2):
        for j in range(5):
            U455[i*5+j, 0:46] = U255[i*5+j, 0:46]
    for j in range(5):
        for k in range(5):
            V455[j*5+k, 0:46] = V255[j*5+k, 0:46]
    for k in range(5):
        for i in range(2):
            W455[k*4+i, 0:46] = W255[k*2+i, 0:46]
            
    # Part 2: i=2,3 (T_255)
    for i in range(2):
        for j in range(5):
            U455[(i+2)*5+j, 46:92] = U255[i*5+j, 0:46]
    for j in range(5):
        for k in range(5):
            V455[j*5+k, 46:92] = V255[j*5+k, 0:46]
    for k in range(5):
        for i in range(2):
            W455[k*4+(i+2), 46:92] = W255[k*2+i, 0:46]
            
    return {"U": U455.tolist(), "V": V455.tolist(), "W": W455.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))

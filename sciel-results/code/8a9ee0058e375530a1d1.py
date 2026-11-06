import random
import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Precompute permutations for 4 elements
    perms = list(itertools.permutations([0, 1, 2, 3]))
    
    def check_q3_in_submatrix(mat, rows, cols):
        # mat is 8x8 numpy array
        # rows, cols are indices 0-7
        sub = mat[np.ix_(rows, cols)]
        # zeros are where mat is 0
        # We want to know if zeros contain a perfect matching
        # i.e. exists perm such that sub[i, perm[i]] == 0 for all i
        for perm in perms:
            if all(sub[i, perm[i]] == 0 for i in range(4)):
                return True
        return False

    def find_bad_submatrices(mat):
        bad = []
        # Iterate over all 4-combinations of rows and cols
        for r1 in range(5):
            for r2 in range(r1+1, 6):
                for r3 in range(r2+1, 7):
                    for r4 in range(r3+1, 8):
                        rows = (r1, r2, r3, r4)
                        for c1 in range(5):
                            for c2 in range(c1+1, 6):
                                for c3 in range(c2+1, 7):
                                    for c4 in range(c3+1, 8):
                                        cols = (c1, c2, c3, c4)
                                        if check_q3_in_submatrix(mat, rows, cols):
                                            bad.append((rows, cols))
        return bad

    best_mat = None
    best_count = 0
    
    for restart in range(200):
        if time.time() > deadline - 5:
            break
            
        # Start with all 1s
        mat = np.ones((8, 8), dtype=int)
        
        # Phase 1: Remove edges to destroy all Q3s
        steps = 0
        while steps < 100:
            bads = find_bad_submatrices(mat)
            if not bads:
                break
            # Pick a random bad submatrix
            rows, cols = random.choice(bads)
            # Pick a random 1 in this submatrix and set to 0
            sub = mat[np.ix_(rows, cols)]
            ones = np.argwhere(sub == 1)
            if len(ones) == 0:
                break
            r_idx, c_idx = random.choice(ones)
            mat[rows[r_idx], cols[c_idx]] = 0
            steps += 1
            
        # Phase 2: Local Search
        zeros = np.argwhere(mat == 0)
        random.shuffle(zeros)
        
        for r, c in zeros:
            if time.time() > deadline - 5:
                break
            mat[r, c] = 1
            bad_found = False
            other_rows = [i for i in range(8) if i != r]
            other_cols = [j for j in range(8) if j != c]
            
            for r1 in range(4):
                for r2 in range(r1+1, 5):
                    for r3 in range(r2+1, 6):
                        rows = (r, other_rows[r1], other_rows[r2], other_rows[r3])
                        for c1 in range(4):
                            for c2 in range(c1+1, 5):
                                for c3 in range(c2+1, 6):
                                    cols = (c, other_cols[c1], other_cols[c2], other_cols[c3])
                                    if check_q3_in_submatrix(mat, rows, cols):
                                        bad_found = True
                                        break
                            if bad_found: break
                        if bad_found: break
                    if bad_found: break
                if bad_found: break
            if bad_found:
                mat[r, c] = 0
        
        count = np.sum(mat)
        if count > best_count:
            best_count = count
            best_mat = mat.copy()

    if best_mat is None:
        return []
        
    edges = []
    for i in range(8):
        for j in range(8):
            if best_mat[i, j] == 1:
                edges.append([i, j + 8])
                
    return edges

            for k in range(p):
                c = j * p + k
                if c < r:
                    V[j * p + k][c] = 1
        patterns.append((U, V))
    
    # Pattern 2: Block structure - U[i,j,c] = 1 if c in block for (i,j)
    # Try various block assignments
    for perm_seed in range(min(50, max(1, (int((deadline - time.time()) * 10) % 100))):
        if time.time() > deadline:
            break
        
        # Try: assign each (i,j) pair a unique column from 0..r-1
        # U[i*m+j, c] = 1 if c == assigned_col(i,j)
        # V[j*p+k, c] = 1 if c == assigned_col(i,j) for some i... 
        # But V doesn't depend on i, so this won't work directly.
        

    # Find a valid center O
    idx = np.argsort(hs)[::-1] # Indices of largest supports (tightest constraints)
    
    # Try intersections of pairs of the top 3 constraints
    valid_O = None
    for i1 in range(3):
        for i2 in range(i1+1, 3):
            n1 = nks[idx[i1]]
            n2 = nks[idx[i2]]
            b1 = hs[idx[i1]] - L * S3 / 2
            b2 = hs[idx[i2]] - L * S3 / 2
            
            A = np.array([n1, n2])
            b = np.array([b1, b2])
            try:
                O_candidate = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
                
            # Check if O_candidate satisfies all 6 constraints
            valid = True
            for k in range(6):
                if nks[k] @ O_candidate < hs[k] - L * S3 / 2 - 1e-9:
                    valid = False
                    break
            if valid:
                valid_O = O_candidate
                break
        if valid_O is not None:
            break
            
    if valid_O is None:
        # Fallback to centroid
        valid_O = np.mean(centers, axis=0)
        
    return L, valid_O, phi

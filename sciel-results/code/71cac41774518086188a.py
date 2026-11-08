    def try_local_search(rank, max_restarts=50, max_steps=50000):
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                return None
            
            U = np.random.choice([-1, 0, 1], size=(9, rank))
            V = np.random.choice([-1, 0, 1], size=(9, rank))
            W = np.random.choice([-1, 0, 1], size=(9, rank))
            
            # Calculate initial E and SSE
            E = np.zeros((3, 3, 3))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        E[i,j,k] = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*3+i, :])
                        
            SSE = np.sum((1.0 - E)**2)
            
            if SSE == 0:
                return (U, V, W)
                
            for step in range(max_steps):
                if time.time() - start_time > time_limit:
                    return None
                    
                mat_idx = random.randint(0, 2)
                row_idx = random.randint(0, 8)
                col_idx = random.randint(0, rank - 1)
                
                old_val = 0
                delta = 0
                affected_E_indices = []
                
                if mat_idx == 0:
                    i = row_idx // 3
                    j = row_idx % 3
                    old_val = U[row_idx, col_idx]
                    candidates = [0, 1, -1]
                    candidates.remove(old_val)
                    # Pick a random candidate? Or try all?
                    # Trying all is better for local search
                    for new_val in candidates:
                        delta = new_val - old_val
                        if delta == 0: continue
                        
                        # Calculate change in E
                        # E[i,j,k] changes by delta * V[j,k,col_idx] * W[k,i,col_idx]
                        # k is 0..2
                        old_SSE = SSE
                        old_E_vals = [0,0,0]
                        for k in range(3):
                            old_E_vals[k] = E[i,j,k]
                            dE = delta * V[j*3+k, col_idx] * W[k*3+i, col_idx]
                            E[i,j,k] += dE
                            SSE += (1.0 - E[i,j,k])**2 - (1.0 - old_E_vals[k])**2
                        
                        if SSE < old_SSE:
                            # Accept
                            U[row_idx, col_idx] = new_val
                            if SSE == 0:
                                return (U, V, W)
                            break
                        else:
                            # Revert
                            for k in range(3):
                                E[i,j,k] = old_E_vals[k]
                            SSE = old_SSE
                            
                elif mat_idx == 1:
                    j = row_idx // 3
                    k = row_idx % 3
                    old_val = V[row_idx, col_idx]
                    candidates = [0, 1, -1]
                    candidates.remove(old_val)
                    for new_val in candidates:
                        delta = new_val - old_val
                        if delta == 0: continue
                        
                        old_SSE = SSE
                        old_E_vals = [0,0,0]
                        # V[j,k,r] affects E[i,j,k] for all i
                        for i in range(3):
                            old_E_vals[i] = E[i,j,k]
                            dE = delta * U[i*3+j, col_idx] * W[k*3+i, col_idx]
                            E[i,j,k] += dE
                            SSE += (1.0 - E[i,j,k])**2 - (1.0 - old_E_vals[i])**2
                            
                        if SSE < old_SSE:
                            V[row_idx, col_idx] = new_val
                            if SSE == 0:
                                return (U, V, W)
                            break
                        else:
                            for i in range(3):
                                E[i,j,k] = old_E_vals[i]
                            SSE = old_SSE
                            
                else:
                    k = row_idx // 3
                    i = row_idx % 3
                    old_val = W[row_idx, col_idx]
                    candidates = [0, 1, -1]
                    candidates.remove(old_val)
                    for new_val in candidates:
                        delta = new_val - old_val
                        if delta == 0: continue
                        
                        old_SSE = SSE
                        old_E_vals = [0,0,0]
                        # W[k,i,r] affects E[i,j,k] for all j
                        for j in range(3):
                            old_E_vals[j] = E[i,j,k]
                            dE = delta * U[i*3+j, col_idx] * V[j*3+k, col_idx]
                            E[i,j,k] += dE
                            SSE += (1.0 - E[i,j,k])**2 - (1.0 - old_E_vals[j])**2
                            
                        if SSE < old_SSE:
                            W[row_idx, col_idx] = new_val
                            if SSE == 0:
                                return (U, V, W)
                            break
                        else:
                            for j in range(3):
                                E[i,j,k] = old_E_vals[j]
                            SSE = old_SSE
                            
        return None

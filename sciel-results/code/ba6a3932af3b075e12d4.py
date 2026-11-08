    def try_reduce_rank(U, V, W, rank, max_removals=5):
        best_U, best_V, best_W, best_rank = U, V, W, rank
        
        # Try removing single terms
        for r_remove in range(rank):
            if time.time() > deadline:
                break
            
            U_new = np.delete(U, r_remove, axis=1)
            V_new = np.delete(V, r_remove, axis=1)
            W_new = np.delete(W, r_remove, axis=1)
            
            if verify_fast(U_new, V_new, W_new, rank - 1):
                best_U, best_V, best_W, best_rank = U_new, V_new, W_new, rank - 1
                break
        
        if best_rank < rank:
            return best_U, best_V, best_W, best_rank

        # Try removing pairs of terms
        for r1 in range(rank):
            if time.time() > deadline:
                break
            for r2 in range(r1 + 1, rank):
                U_new = np.delete(U, [r1, r2], axis=1)
                V_new = np.delete(V, [r1, r2], axis=1)
                W_new = np.delete(W, [r1, r2], axis=1)
                
                if verify_fast(U_new, V_new, W_new, rank - 2):
                    best_U, best_V, best_W, best_rank = U_new, V_new, W_new, rank - 2
                    return best_U, best_V, best_W, best_rank
                    
        return best_U, best_V, best_W, best_rank

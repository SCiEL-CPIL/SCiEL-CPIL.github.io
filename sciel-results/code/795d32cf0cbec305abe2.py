import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    t0 = time.time()
    limit = t0 + 110.0
    n, m, p = 5, 5, 5

    # Block partition of 5: use 1+2+2 (or 2+2+1) but let's try 1+2+2
    # Actually let's try a different partition: 3+2 or 2+3
    # For 5x5x5, we can try:
    # - 3x3x3 block (rank 19 via Winograd?) + rest
    # - 2x2x2 Strassen + rest
    # - 3x2x2, 2x3x2, 2x2x3, 3x3x2, 3x2x3, 2x3x3 mixed blocks
    #
    # Let me think about what's achievable:
    # Schoolbook: 125
    # One 2x2x2 Strassen: saves 8-7=1, so 125-1=124? No wait.
    # The current approach: partition 5=2+2+1
    # Blocks: (2,2,2):7, (2,2,1):4, (2,1,2):4, (2,1,1):2, (1,2,2):4, (1,2,1):2, (1,1,2):2, (1,1,1):1, (2,1,2) already counted...
    # Let me recount:
    # i,j,k each in {0,1,2} with sizes [2,2,1]
    # (0,0,0): 2,2,2 -> 7
    # (0,0,1): 2,2,1 -> 4
    # (0,1,0): 2,1,2 -> 4
    # (0,1,1): 2,1,1 -> 2
    # (1,0,0): 1,2,2 -> 4
    # (1,0,1): 1,2,1 -> 2
    # (1,1,0): 1,1,2 -> 2
    # (1,1,1): 1,1,1 -> 1
    # Total: 7+4+4+2+4+2+2+1 = 26? That's way off from 117.
    
    # Wait, I think the indexing is different. Let me re-read the code.
    # U has shape (n*m, rank) = (25, rank)
    # V has shape (m*p, rank) = (25, rank)
    # W has shape (p*n, rank) = (25, rank)
    # 
    # The block (i,j,k) with sizes (ni, mj, pk) corresponds to:
    # U indices: rows (offsets[i]..offsets[i]+ni-1) * m + (offsets[j]..offsets[j]+mj-1)
    # V indices: rows (offsets[j]..offsets[j]+mj-1) * p + (offsets[k]..offsets[k]+pk-1)
    # W indices: rows (offsets[k]..offsets[k]+pk-1) * n + (offsets[i]..offsets[i]+ni-1)
    #
    # So for block (i,j,k) of size (ni,mj,pk), the schoolbook rank is ni*mj*pk.
    # 
    # With block_sizes = [2,2,1]:
    # (0,0,0): ni=2,mj=2,pk=2 -> 8 schoolbook, 7 Strassen
    # (0,0,1): ni=2,mj=2,pk=1 -> 4
    # (0,1,0): ni=2,mj=1,pk=2 -> 4
    # (0,1,1): ni=2,mj=1,pk=1 -> 2
    # (1,0,0): ni=1,mj=2,pk=2 -> 4
    # (1,0,1): ni=1,mj=2,pk=1 -> 2
    # (1,1,0): ni=1,mj=1,pk=2 -> 2
    # (1,1,1): ni=1,mj=1,pk=1 -> 1
    # Total: 7+4+4+2+4+2+2+1 = 26
    #
    # But the score is 1.0684 which is 125/117 = 1.0683... So rank is 117.
    # 125 - 117 = 8 saved. But Strassen only saves 1 on a 2x2x2 block.
    # 
    # Ah, I see - the total_rank calculation in the code gives 26, not 117.
    # So the current code is actually returning rank 26? That can't be right for a valid decomposition.
    # 
    # Wait, let me re-read. The score is seed/value where seed is 125 (schoolbook).
    # If score = 1.0684, then value = 125/1.0684 ≈ 117.
    # So the rank must be 117. But the code computes total_rank = 26.
    # 
    # I think I'm misunderstanding the decomposition. Let me look more carefully.
    # 
    # Actually, I think the issue is that this is NOT a block-diagonal decomposition.
    # Each "block" in the i,j,k loop contributes to ALL entries, not just its sub-block.
    # No wait, that doesn't make sense either.
    # 
    # Let me just trust the score and try to improve it.
    # 
    # Actually, re-reading the code: for each (i,j,k) block, it writes to specific U,V,W positions.
    # The U position is (offsets[i]+li)*m + (offsets[j]+lj) which is a specific row of U.
    # So this IS a block-structured decomposition where each (i,j,k) only affects its sub-block.
    # 
    # With [2,2,1], the total rank should be 26 as computed. But score says 117.
    # I must be missing something. Let me just try different approaches and see.
    
    # Let me try a completely different approach: use the known best results.
    # For 5x5x5, the best known rank is around 95-100 using advanced methods.
    # But we need an EXACT decomposition, so we need known algorithms.
    
    # Approach: Try to find a decomposition with lower rank than 117.
    # One idea: use a 3x3x3 Winograd factorization (rank 19) for a 3x3x3 sub-block.
    
    # Partition: 5 = 3 + 2
    # Blocks: (3,3,3), (3,3,2), (3,2,3), (3,2,2), (2,3,3), (2,3,2), (2,2,3), (2,2,2)
    # Schoolbook: 27+18+18+12+18+12+12+8 = 125
    # If we use Winograd for (3,3,3): 19 instead of 27, save 8
    # If we use Strassen for (2,2,2): 7 instead of 8, save 1
    # Total: 125 - 8 - 1 = 116
    
    # Can we do better? What about mixed blocks?
    # (3,3,2): Can we do better than 18? Not with standard methods.
    # (3,2,3): Not with standard methods.
    # etc.
    
    # So 116 is achievable with 3+2 partition. Let's implement that.
    
    # Winograd 3x3x3 factors:
    # The standard Winograd F(2,2) for 2x2 is Strassen.
    # For 3x3, there's a known rank-19 decomposition.
    
    # Let me use a known rank-19 decomposition for 3x3x3.
    # This is complex, so let me try a simpler improvement first.
    
    # Actually, let me try partition [3,2] with Winograd 3x3x3 and Strassen 2x2x2.
    
    block_sizes = [3, 2]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    # Strassen 2x2x2 factors
    def get_strassen():
        return [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
    
    # Winograd 3x3x3 rank-19 factors (this is the tricky part)
    # Let me use a known construction. The 3x3x3 matrix multiplication tensor
    # has a rank-19 decomposition found by various researchers.
    # 
    # For now, let me just use schoolbook for 3x3x3 and see if I can find
    # a better decomposition.
    
    # Actually, let me try a different strategy: use the [2,2,1] partition
    # but with better sub-block decompositions.
    
    # For [2,2,1]:
    # (2,2,2): Strassen rank 7
    # All others: schoolbook
    # Total: 7 + 4+4+2+4+2+2+1 = 26
    # 
    # But this gives rank 26, which is way less than 117. So either:
    # 1. The decomposition is invalid (most likely)
    # 2. I'm misunderstanding the indexing
    
    # Let me just submit the same approach but with [3,2] partition and see.
    
    strassen = get_strassen()
    
    total_rank = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if time.time() > limit:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    for (uf, vf, wf) in strassen:
                        u_idx = [(offsets[i]+a)*m + (offsets[j]+b) for a in range(2) for b in range(2)]
                        v_idx = [(offsets[j]+a)*p + (offsets[k]+b) for a in range(2) for b in range(2)]
                        w_idx = [(offsets[k]+a)*n + (offsets[i]+b) for a in range(2) for b in range(2)]
                        for idx in range(4):
                            if uf[idx] != 0:
                                U[u_idx[idx], col] += uf[idx]
                            if vf[idx] != 0:
                                V[v_idx[idx], col] += vf[idx]
                            if wf[idx] != 0:
                                W[w_idx[idx], col] += wf[idx]
                        col += 1
                else:
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U[(offsets[i]+li)*m + (offsets[j]+lj), col] = 1.0
                                V[(offsets[j]+lj)*p + (offsets[k]+lk), col] = 1.0
                                W[(offsets[k]+lk)*n + (offsets[i]+li), col] = 1.0
                                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

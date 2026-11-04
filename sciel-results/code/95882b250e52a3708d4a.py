    def expand_v(v_in, blocks):
        v = zero_v()
        # V is indexed by j*p + k. m=4, p=7.
        # Block 0: j in {0,1}, k in {0,1}. Indices: 0, 1, 7, 8
        # Block 1: j in {0,1}, k in {2,3}. Indices: 2, 3, 9, 10
        # Block 2: j in {2,3}, k in {0,1}. Indices: 14, 15, 21, 22
        # Block 3: j in {2,3}, k in {2,3}. Indices: 16, 17, 23, 24
        
        # Map (block_off, inner_idx) to global index
        # inner_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # block 0: j_base=0, k_base=0
        # block 1: j_base=0, k_base=2
        # block 2: j_base=2, k_base=0
        # block 3: j_base=2, k_base=2
        
        for block_off, sign in blocks:
            j_base = [0, 0, 2, 2][block_off]
            k_base = [0, 2, 0, 2][block_off]
            for k_inner in range(4):
                if v_in[k_inner] != 0:
                    # inner k_inner 0 -> j_offset 0, k_offset 0
                    # inner k_inner 1 -> j_offset 0, k_offset 1
                    # inner k_inner 2 -> j_offset 1, k_offset 0
                    # inner k_inner 3 -> j_offset 1, k_offset 1
                    j_off = [0, 0, 1, 1][k_inner]
                    k_off = [0, 1, 0, 1][k_inner]
                    j = j_base + j_off
                    k = k_base + k_off
                    idx = j * p + k
                    v[idx] += sign * v_in[k_inner]
        return v

def construct():
    # Precompute homogeneity for all masks
    is_hom = [False] * (1 << N)
    for mask in range(1, 1 << N):
        # Check clique
        ok_c = True
        m = mask
        while m:
            v = (m & -m).bit_length() - 1
            m ^= (1 << v)
            if m & ~adj[v]:
                ok_c = False
                break
        if ok_c:
            is_hom[mask] = True
        else:
            # Check independent
            ok_i = True
            m = mask
            while m:
                v = (m & -m).bit_length() - 1
                m ^= (1 << v)
                if m & adj[v]:
                    ok_i = False
                    break
            if ok_i:
                is_hom[mask] = True

    def find_partition(k):
        def backtrack(remaining, parts):
            if remaining == 0:
                return True
            
            v = (remaining & -remaining).bit_length() - 1
            v_bit = 1 << v
            
            # Try adding v to existing parts
            for i in range(len(parts)):
                p = parts[i]
                new_p = p | v_bit
                if is_hom[new_p]:
                    parts[i] = new_p
                    if backtrack(remaining ^ v_bit, parts):
                        return True
                    parts[i] = p
            
            # Try adding v to a new part
            if len(parts) < k:
                parts.append(v_bit)
                if backtrack(remaining ^ v_bit, parts):
                    return True
                parts.pop()
            
            return False
        
        parts = []
        if backtrack(FULL, parts):
            # Convert masks to lists
            result = []
            for mask in parts:
                verts = []
                m = mask
                while m:
                    v = (m & -m).bit_length() - 1
                    m ^= (1 << v)
                    verts.append(v)
                result.append(verts)
            return result
        return None

    for k in range(1, N + 1):
        res = find_partition(k)
        if res:
            return res

    # Fallback (should not be reached)
    return [[v] for v in range(N)]

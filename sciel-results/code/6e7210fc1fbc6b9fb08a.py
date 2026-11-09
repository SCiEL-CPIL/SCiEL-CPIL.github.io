# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580
    
    # Generate candidates with norm^2 = 4
    # 4 non-zero entries, each +/- 1
    # C(11, 4) * 2^4 = 330 * 16 = 5280
    
    candidates = []
    for support in range(1 << d):
        # Check if support has exactly 4 bits set
        if bin(support).count('1') != 4:
            continue
        # Get indices
        indices = [i for i in range(d) if (support >> i) & 1]
        # Generate all sign combinations
        for signs in range(16):
            v = np.zeros(d, dtype=np.int64)
            for k, idx in enumerate(indices):
                if (signs >> k) & 1:
                    v[idx] = 1
                else:
                    v[idx] = -1
            candidates.append(v)
            
    # Shuffle candidates
    random.shuffle(candidates)
    
    selected = []
    
    # Pre-compute supports and sign vectors for faster checking?
    # Or just use the array directly.
    # With 5280 candidates and potentially ~2000 selected, 
    # 5280 * 2000 = 10.5M checks.
    # Each check involves finding intersection of supports.
    # Supports are 11-bit integers.
    
    # Let's store (vector, support_int)
    cand_data = []
    for v in candidates:
        support_int = 0
        for i in range(d):
            if v[i] != 0:
                support_int |= (1 << i)
        cand_data.append((v, support_int))
        
    for v, supp_v in cand_data:
        # Check against all selected
        compatible = True
        for u, supp_u in selected:
            inter = supp_u & supp_v
            inter_size = bin(inter).count('1')
            
            if inter_size == 4:
                # Same support
                if np.array_equal(u, v):
                    compatible = False
                    break
                # else compatible (distinct vectors with same support have dot <= 2)
            elif inter_size == 3:
                # Check dot product on intersection
                # Extract values on intersection
                # This is a bit slow if done naively.
                # We can precompute? Or just loop 3 times.
                dot = 0
                for i in range(d):
                    if (inter >> i) & 1:
                        dot += u[i] * v[i]
                if dot == 3:
                    compatible = False
                    break
            # else (0, 1, 2) compatible
            
        if compatible:
            selected.append((v, supp_v))
            
        # Time check
        if time.time() > deadline:
            break

    # Convert to array
    if len(selected) >= 2:
        points = np.array([s[0] for s in selected], dtype=np.int64)
        return np.around(points).astype(np.int64)
    else:
        # Fallback
        points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
        return np.around(points).astype(np.int64)

# EVOLVE-BLOCK-END

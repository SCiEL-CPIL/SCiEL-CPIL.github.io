def _try_reverse(chain, occupied, pos_to_idx, sequence, i, j):
    # Reverse chain[i..j]
    # Check self-avoidance
    # New positions are chain[j], chain[j-1], ..., chain[i]
    # The set of positions is the same, just permuted.
    # So we just need to check if the new adjacency is valid (it is, it's a chain)
    # and if any new position overlaps with an outside node.
    # Wait, the positions are the SAME set.
    # So self-avoidance is guaranteed IF the original was self-avoiding?
    # No!
    # Original: ... A - B - C - D ...
    # Reverse B-C: ... A - C - B - D ...
    # Positions: A, C, B, D.
    # Are C and B adjacent to A or D correctly?
    # A is at p_A. C is at p_C. B is at p_B. D is at p_D.
    # Original bonds: (A,B), (B,C), (C,D).
    # New bonds: (A,C), (C,B), (B,D).
    # We need to check if (A,C) is a unit step? No, A and C were distance 2 apart in the chain, but geometrically they might be far apart.
    # So 2-opt is NOT a simple permutation of existing positions.
    # It creates NEW bonds.
    # So we must check if the new bonds are unit steps.
    # And if the new configuration is self-avoiding.
    # Since the set of occupied sites is the same, self-avoidance is preserved.
    # We ONLY need to check if the new bonds are valid unit steps.
    # Bond 1: chain[i-1] to chain[j] (new first of segment). Must be unit step.
    # Bond 2: chain[j] to chain[j-1]. Already unit step (was chain[j-1] to chain[j]).
    # ...
    # Bond k: chain[i+1] to chain[i]. Already unit step.
    # Bond k+1: chain[i] to chain[i+1] (new last of segment to next). Must be unit step.
    
    # So, 2-opt is valid if:
    # 1. dist(chain[i-1], chain[j]) == 1
    # 2. dist(chain[i], chain[i+1]) == 1 (if i+1 < n)
    
    # If valid, we just update the chain list and pos_to_idx.
    # Energy update:
    # The set of occupied sites is unchanged.
    # The bonds are changed.
    # Contacts depend on proximity, not bonds.
    # So the set of H-H contacts is UNCHANGED?
    # No! The contacts are between non-bonded neighbors.
    # The set of positions is the same. So the geometric proximity is the same.
    # So the energy is UNCHANGED?
    # Yes! In a lattice model, if you permute the residues along the chain but keep them on the same sites, the set of occupied sites is identical.
    # Therefore, the set of H-H contacts (which depends only on which sites are occupied by H) is identical.
    # So 2-opt moves do NOT change the energy.
    # They are neutral moves.
    # They are useful for exploring conformational space (changing the topology of the chain) without changing energy.
    # This is great for escaping local minima in the "energy landscape" defined by the chain topology.
    
    # So, we can accept 2-opt moves with high probability (e.g., 0.5 or 1.0) if they are valid topologically.

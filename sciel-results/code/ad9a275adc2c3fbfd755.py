<<<<<<< SEARCH
def _greedy_init():
    """Greedy construction: build a dense comparable structure by adding masks that maximize marginal gain."""
    all_masks = list(range(1 << N_VERT))
    random.seed(42)
    
    # Start with a few well-chosen seeds to create a base structure
    # Use masks of varying sizes to create comparability
    selected = []
    
    # Add empty set and full set as anchors
    selected.append(0)
    selected.append((1 << N_VERT) - 1)
    
    # Greedily add masks that maximize comparable pairs with existing set
    for _ in range(M - 2):
        best_mask = None
        best_gain = -1
        
        # Sample a subset of candidates for speed (not all 4096)
        candidates = random.sample(all_masks, min(200, len(all_masks)))
        
        for mask in candidates:
            if mask in selected:
                continue
            # Count how many existing sets are comparable with this mask
            gain = 0
            for existing in selected:
                if (mask & existing) == mask or (mask & existing) == existing:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
        
        if best_mask is not None:
            selected.append(best_mask)
    
    return selected
=======
def _greedy_init():
    """Greedy construction using chains from random permutations."""
    # Use a fixed seed for reproducibility of the initial structure,
    # but ILS will provide stochastic exploration.
    random.seed(42)
    
    selected = set()
    chains = []
    
    # Create 4 random permutations
    for _ in range(4):
        perm = list(range(N_VERT))
        random.shuffle(perm)
        chains.append(perm)
        
    # Take 10 sets from each chain
    for chain in chains:
        current_mask = 0
        for k in range(N_VERT):
            current_mask |= (1 << chain[k])
            if current_mask not in selected:
                selected.add(current_mask)
                if len(selected) == M:
                    break
        if len(selected) == M:
            break
            
    # If we don't have 40, fill with random masks
    if len(selected) < M:
        all_masks = list(range(1 << N_VERT))
        random.shuffle(all_masks)
        for mask in all_masks:
            if mask not in selected:
                selected.add(mask)
                if len(selected) == M:
                    break
                    
    return list(selected)
>>>>>>> REPLACE


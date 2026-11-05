import time
import itertools

def construct():
    start_time = time.time()
    deadline = 120  # seconds
    
    # Points 0..14 correspond to 1..15 in binary (4-bit)
    # Point 15 is the extra point.
    
    # Generate all triples in STS(15) (lines of PG(3,2))
    # A line is {u, v, u^v} for u != v, u != 0, v != 0, u != v
    # We use integers 1..15 to represent points 0..14
    # Mapping: point i in output is i for i in 0..14?
    # Let's map output point p (0..14) to vector p+1 (1..15).
    # Output point 15 is 15.
    
    triples = []
    seen_pairs = set()
    
    # Generate all lines
    for u in range(1, 16):
        for v in range(u + 1, 16):
            w = u ^ v
            if w == 0 or w == u or w == v:
                continue
            # Sort to identify unique triple
            t = tuple(sorted((u, v, w)))
            if t not in seen_pairs:
                triples.append(list(t))
                # Add pairs to seen to avoid duplicates? 
                # In STS, each pair is in exactly one triple.
                # So iterating u<v and w is unique enough if we check t.
                seen_pairs.add(t)
                
    # Convert to output indices 0..14
    # Triple (u, v, w) becomes (u-1, v-1, w-1)
    sts_triples = [ [x-1 for x in t] for t in triples ]
    
    # Identify the 15 hemispheres (caps of size 8)
    # A hemisphere is defined by a non-zero linear functional L.
    # L(x) = dot(x, mask) mod 2 = 1
    # Points are 1..15.
    hemispheres = []
    for mask in range(1, 16):
        cap = []
        for x in range(1, 16):
            # Check if dot(x, mask) is 1
            # In Python, use bit_count of x & mask
            if (x & mask).bit_count() % 2 == 1:
                cap.append(x)
        # Convert to 0..14 indices
        cap_indices = [c-1 for c in cap]
        hemispheres.append(set(cap_indices))
        
    # We want to select pairs (a, b) from 0..14
    # For each pair, find which hemispheres contain both a and b.
    # We want to cover all 15 hemispheres.
    
    # Precompute coverage for each pair
    # Pairs are (a, b) with a < b
    pair_coverage = {}
    for a in range(15):
        for b in range(a+1, 15):
            covering_hems = []
            for i, h in enumerate(hemispheres):
                if a in h and b in h:
                    covering_hems.append(i)
            pair_coverage[(a, b)] = set(covering_hems)
            
    # Greedy set cover
    uncovered = set(range(15))
    selected_pairs = []
    
    while uncovered:
        if time.time() - start_time > deadline - 5:
            break
        best_pair = None
        best_count = 0
        for pair, cov in pair_coverage.items():
            if not cov:
                continue
            count = len(cov & uncovered)
            if count > best_count:
                best_count = count
                best_pair = pair
        if best_pair is None:
            break
        selected_pairs.append(best_pair)
        uncovered -= pair_coverage[best_pair]
        
    # Construct final system
    # Start with STS triples
    final_triples = [t[:] for t in sts_triples]
    
    # Remove triples corresponding to selected pairs
    # For pair (a, b), find the STS triple containing a and b.
    # In STS, it is unique.
    for a, b in selected_pairs:
        # Find triple containing a, b
        for i, t in enumerate(final_triples):
            if a in t and b in t:
                # Remove it
                final_triples.pop(i)
                break
        
        # Add new triple {15, a, b}
        final_triples.append([15, a, b])
        
    # Verify linearity
    # Any two triples share at most 1 point
    is_valid = True
    for i in range(len(final_triples)):
        for j in range(i+1, len(final_triples)):
            intersection = set(final_triples[i]) & set(final_triples[j])
            if len(intersection) > 1:
                is_valid = False
                break
        if not is_valid:
            break
            
    if not is_valid:
        # Fallback to STS(15) + isolated 15
        return sts_triples + [] # 15 is isolated, so no triples added.
        # Wait, if we return sts_triples, 15 is isolated.
        # Independence number is 9.
        
    # Calculate independence number
    # We can use a simple search or heuristic.
    # Since 16 points, we can check all subsets? 2^16 = 65536.
    # For each subset, check if it contains a triple.
    # This is fast.
    
    def has_triple(subset_mask, triples):
        # subset_mask is an integer bitmask
        for t in triples:
            # Check if all points in t are in subset_mask
            # t is [a, b, c]
            if (subset_mask >> t[0]) & 1:
                if (subset_mask >> t[1]) & 1:
                    if (subset_mask >> t[2]) & 1:
                        return True
        return False
        
    max_ind = 0
    # Iterate over all subsets
    # We can prune: if size <= max_ind, skip
    for mask in range(1 << 16):
        size = mask.bit_count()
        if size <= max_ind:
            continue
        if not has_triple(mask, final_triples):
            max_ind = size
            
    return final_triples

# EVOLVE-BLOCK-END

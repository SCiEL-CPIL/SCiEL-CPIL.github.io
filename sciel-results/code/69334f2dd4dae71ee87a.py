   def has_q3_fast(adj_bits, n=16):
       # Precompute masks for all 8-subsets? No.
       # Just iterate.
       # To speed up, check if any vertex has degree < 3 in the subset.
       # We can precompute the neighborhood of each vertex as a bitmask.
       
       # Iterate 8-subsets
       for subset in combinations(range(n), 8):
           s = list(subset)
           # Quick check: min degree in induced subgraph
           # We can check this by:
           # For each v in s, count bits in adj_bits[v] & mask(s)
           # If count < 3, skip.
           
           # To avoid creating mask(s) repeatedly, we can compute it once.
           mask_s = 0
           for v in s:
               mask_s |= (1 << v)
               
           min_deg = 100
           for v in s:
               # Remove v from mask to count neighbors
               neighbors = adj_bits[v] & (mask_s ^ (1 << v))
               deg = bin(neighbors).count("1")
               if deg < 3:
                   break
               min_deg = min(min_deg, deg)
           else:
               # All degrees >= 3
               # Now check bipartitions
               for a_set in combinations(s, 4):
                   a = list(a_set)
                   b = [x for x in s if x not in a_set]
                   
                   # Check bipartite condition
                   # For each av in a, missing neighbor in b must be unique
                   missing_map = {}
                   valid = True
                   for av in a:
                       # Find b in b not connected to av
                       # adj_bits[av] & mask_b
                       mask_b = 0
                       for bv in b:
                           mask_b |= (1 << bv)
                       connected = adj_bits[av] & mask_b
                       # missing = mask_b ^ connected
                       # We need exactly 1 bit set in missing
                       missing = mask_b ^ connected
                       if missing == 0 or (missing & (missing - 1)) != 0:
                           valid = False
                           break
                       # missing is a power of 2
                       bv = missing.bit_length() - 1
                       if bv in missing_map:
                           valid = False
                           break
                       missing_map[bv] = av
                   
                   if valid and len(missing_map) == 4:
                       # Found Q3
                       # Return an edge
                       # Pick first edge in Q3
                       # a[0] is connected to 3 in b.
                       # Find one.
                       mask_b = 0
                       for bv in b:
                           mask_b |= (1 << bv)
                       connected = adj_bits[a[0]] & mask_b
                       # Pick lowest set bit
                       bv = (connected & -connected).bit_length() - 1
                       return (True, (a[0], bv))
       
       return (False, None)
   
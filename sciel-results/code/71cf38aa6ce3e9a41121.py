   def _greedy_prune_fast(n, in_bits, deadline):
       active = list(range(n))
       cur_in_bits = in_bits[:]
       
       # We will iteratively try to remove vertices.
       # To avoid O(n^5), we can just loop through vertices.
       # If we remove one, we restart the scan or continue.
       # Since n is small, restarting is fine.
       
       changed = True
       while changed and time.time() < deadline - 5:
           changed = False
           for v in list(active):
               if len(active) <= 5: break
               
               # Check if removing v is valid
               # A 4-set S is bad if it has no dominator in active \ {v}
               # S must not contain v.
               
               valid = True
               # Iterate over all 4-subsets of active \ {v}
               # To speed up, we can precompute masks?
               # Just use combinations.
               
               new_active = [x for x in active if x != v]
               
               # Optimization: If we precomputed dominators for all 4-sets in the FULL set,
               # we can check if any 4-set S (not containing v) has D(S) subset of {v} U (S itself?) No.
               # D(S) is subset of V.
               # We need D(S) \ {v} to intersect (V \ {v} \ S).
               # i.e., D(S) \ {v} is not empty.
               
               # Let's just do the direct check.
               for combo in combinations(new_active, 4):
                   a, b, c, d = combo
                   mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                   common = cur_in_bits[a] & cur_in_bits[b] & cur_in_bits[c] & cur_in_bits[d]
                   # Dominators must be in new_active and not in mask
                   # common contains all dominators in V.
                   # We need at least one in new_active \ mask.
                   # new_active \ mask = (active \ {v}) \ mask.
                   # Since mask is subset of new_active, this is just active \ {v} \ mask.
                   
                   # Let's check if common has any bit set in new_active that is not in mask.
                   # Actually, common & ~mask gives dominators outside the set.
                   # We need one of them to be in new_active.
                   # Since new_active is active \ {v}, and common bits are in active (assuming valid prev state),
                   # we just need to ensure that the dominator is not v.
                   
                   doms = common & ~mask
                   if doms == 0:
                       valid = False
                       break
                   
                   # Check if any dominator is in new_active
                   # doms might contain v?
                   # If v is a dominator, and v is removed, it's not in new_active.
                   # So we need (doms & ~(1 << v)) to have a bit in new_active?
                   # Actually, if v is the ONLY dominator, then doms = (1 << v).
                   # Then doms & ~(1 << v) = 0.
                   # So checking if (doms & ~(1 << v)) != 0 is sufficient?
                   # Wait, doms might contain other vertices not in active?
                   # No, in a valid tournament, dominators are in active.
                   # So doms is subset of active.
                   # If we remove v, we need a dominator in active \ {v}.
                   # So we need (doms & ~(1 << v)) != 0.
                   
                   if (doms & ~(1 << v)) == 0:
                       valid = False
                       break
               
               if valid:
                   # Remove v
                   active = new_active
                   changed = True
                   # Optionally update cur_in_bits? Not strictly needed if we just use the original in_bits
                   # and check against 'active'.
                   # But for the next iteration, 'active' is smaller.
                   # The check `doms & ~(1 << v)` works with the original in_bits.
                   # We just need to ensure we only consider vertices in 'active'.
                   # The combinations(new_active, 4) ensures we only check valid sets.
                   # The condition `(doms & ~(1 << v)) != 0` ensures the dominator is not v.
                   # But what if the dominator is not in `active` at all?
                   # If the tournament was valid before, dominators were in `active`.
                   # After removal, if the only dominator was v, we catch it.
                   # If there was another dominator u, u is in `active` (since we only removed v).
                   # So u is in `new_active`.
                   # So the check is correct.
                   
                   break # Restart the loop over active vertices? Or continue?
                   # Restarting is safer to find the next removable vertex.
                   # But breaking the inner loop and restarting the while loop is effectively restarting.
                   # Let's just `break` the inner for loop, and the `while changed` will loop again.
       
       return len(active), active
   
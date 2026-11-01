      def local_search_improve(block_indices):
          current_blocks = list(block_indices)
          if not current_blocks: return []
          
          cover_counts = [0] * num_t_subsets
          for idx in current_blocks:
              m = masks[idx]
              while m:
                  lsb = m & -m
                  j = lsb.bit_length() - 1
                  cover_counts[j] += 1
                  m ^= lsb
          
          improved = True
          while improved and time.time() < deadline:
              improved = False
              
              # Calculate critical count for each block
              # critical_count[i] = number of t-subsets j in i where cover_counts[j] == 1
              # We only need to check this for blocks we consider removing.
              
              # Sort blocks by "removability" heuristic
              # A block is removable if critical_count == 0.
              # We can find all such blocks quickly.
              
              removable_candidates = []
              for idx in current_blocks:
                  m = masks[idx]
                  is_removable = True
                  while m:
                      lsb = m & -m
                      j = lsb.bit_length() - 1
                      if cover_counts[j] == 1:
                          is_removable = False
                          break
                      m ^= lsb
                  if is_removable:
                      removable_candidates.append(idx)
              
              if not removable_candidates:
                  break
              
              # Randomly pick one to remove? Or all?
              # If we remove all, we might break coverage?
              # No, if critical_count == 0, removing it doesn't uncover any t-subset.
              # So we can remove ALL of them in one pass.
              
              # However, removing one might make another removable?
              # No, removing a block decreases cover_counts.
              # If a t-subset had count 2, and we remove a block covering it, count becomes 1.
              # This makes it *harder* to remove other blocks covering it.
              # So removing blocks makes others *less* likely to be removable.
              # Therefore, we should remove as many as possible in one pass?
              # Wait.
              # If I remove A, cover_counts decrease.
              # A t-subset with count 2 becomes 1.
              # A block B covering that t-subset now has critical_count > 0.
              # So B becomes NOT removable.
              # So removing blocks can "poison" other blocks.
              # Thus, the order matters.
              # To maximize removals, we should remove blocks that cover the FEWEST critical t-subsets?
              # No, if critical_count is 0, it's safe.
              # But if I remove A (safe), I might make B unsafe.
              # If I removed B first (safe), I might make A unsafe.
              # This is a complex dependency.
              
              # Heuristic: Remove blocks with the LOWEST "impact" on other blocks' criticality.
              # Or just random order, as before, but batch the removals?
              # If I identify a set of mutually non-interfering removable blocks, I can remove them all.
              # Two blocks A and B interfere if they share a t-subset with count 2.
              
              # Simpler: Just do one removal at a time, random order.
              # But we can optimize the check.
              
              random.shuffle(removable_candidates)
              for idx in removable_candidates:
                  # Re-check if still removable (in case previous removals in this loop made it not)
                  m = masks[idx]
                  still_removable = True
                  while m:
                      lsb = m & -m
                      j = lsb.bit_length() - 1
                      if cover_counts[j] == 1:
                          still_removable = False
                          break
                      m ^= lsb
                  
                  if still_removable:
                      # Remove
                      m = masks[idx]
                      while m:
                          lsb = m & -m
                          j = lsb.bit_length() - 1
                          cover_counts[j] -= 1
                          m ^= lsb
                      current_blocks.remove(idx)
                      improved = True
          
          return current_blocks
      
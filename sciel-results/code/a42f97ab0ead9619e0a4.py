      def has_q3(adj_mask, s):
          # s is tuple of 8
          # 70 partitions
          # Precomputed partitions of indices 0..7
          for a_idx, b_idx in PARTITIONS:
              # a_idx, b_idx are tuples of 4 indices
              # Check missing edges
              # Count missing edges first?
              # If < 4, skip.
              miss_count = 0
              # Build list of missing edges?
              # Just check matchings directly?
              # If we check 24 matchings, and each requires 4 bit checks.
              # 96 bit checks.
              # If we count first, it's 16 bit checks.
              # If count < 4, we save 80 checks.
              # If count >= 4, we do 96 checks.
              
              # Let's just do the matching check, it's simple.
              for perm in PERMS:
                  ok = True
                  for i in range(4):
                      u = s[a_idx[i]]
                      v = s[b_idx[perm[i]]]
                      if (adj_mask[u] >> v) & 1:
                          ok = False
                          break
                  if ok:
                      return True
          return False
      
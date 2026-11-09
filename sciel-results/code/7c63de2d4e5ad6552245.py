        def try_change_residue(self, idx, new_r):
            p = self.p[idx]
            old_r = self.r[idx]
            
            if old_r == new_r:
                return self.get_L(), True
            
            current_L = self.get_L()
            
            # Check for holes
            # Risk set: n in [1, current_L] such that n % p == old_r and n % p != new_r
            # Since n % p == old_r implies n % p != new_r (if old_r != new_r),
            # we just check all n in old_r class in [1, current_L].
            
            old_start = old_r if old_r >= 1 else old_r + p
            
            # Check if any n in old class has count 1
            for n in range(old_start, current_L + 1, p):
                if self.counts[n] == 1:
                    # This move would uncover n
                    return self.get_L(), False
            
            # Move is safe. Apply it.
            # Update counts
            # Decrement old
            for n in range(old_start, current_L + 1, p):
                self.counts[n] -= 1
            # Note: We only updated up to current_L.
            # What about n > current_L?
            # If n > current_L, it was a gap or beyond.
            # If we uncover it, it doesn't affect L (which is defined by the first gap).
            # But if we *cover* a number > current_L, L might increase.
            # So we need to update counts for n > current_L as well?
            # Yes, to correctly calculate the new L.
            
            # Let's update counts for the whole range [1, L_cap] or at least [1, current_L + p]
            # Actually, to be safe and simple, let's just update [1, L_cap] if p is small?
            # No, that's slow.
            
            # Compromise:
            # We know [1, current_L] is safe (no holes).
            # We update counts for [1, current_L + p] for both old and new.
            # Then we scan from current_L + 1 to find new gap.
            
            # But wait, if we decrement old counts for n > current_L, we might make a gap?
            # No, gaps beyond L don't matter for the "prefix" definition, except that we want to *fill* them.
            # If we remove coverage from a number > L, it doesn't hurt L.
            # If we add coverage to a number > L, it might help L.
            
            # So:
            # 1. Decrement old counts in [1, L_cap] (or just [1, current_L + p] is enough?
            #    If we decrement a number > current_L + p, it doesn't affect the immediate extension.
            #    But it might affect a future extension.
            #    Since we are doing a "move" and then "evaluating", we should keep the state consistent.
            #    So we MUST update all affected counts.
            
            # OK, let's update all affected counts in [1, L_cap].
            # This is O(L/p).
            
            # Decrement old
            for n in range(old_start, self.L_cap + 1, p):
                self.counts[n] -= 1
            # Increment new
            new_start = new_r if new_r >= 1 else new_r + p
            for n in range(new_start, self.L_cap + 1, p):
                self.counts[n] += 1
                
            self.r[idx] = new_r
            
            # Calculate new L
            new_L = current_L
            for i in range(current_L + 1, self.L_cap + 1):
                if self.counts[i] > 0:
                    new_L = i
                else:
                    break
            return new_L, True

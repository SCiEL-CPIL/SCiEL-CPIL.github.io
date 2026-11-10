    def try_move(self, idx, new_pos):
        # ... (copy logic) ...
        
        old_max = self.max_mult
        # ... (remove old areas) ...
        
        # Check if we removed the max
        # We need to know if any removed area was == old_max and its count became 0
        # And if we added any area > old_max
        
        # ... (add new areas) ...
        
        # Update max
        # If we added an area > old_max, new_max is that area.
        # Else, if we removed the old_max and it's gone, we need to find new max.
        # Else, new_max is old_max.
        
        # This requires tracking.
        # Given K is small, max(counter.values()) is fine.

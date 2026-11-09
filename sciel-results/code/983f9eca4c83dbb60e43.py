    def backtrack(d):
        if not active:
            return True, None
        
        # Check for contradiction in active clauses with size 0? 
        # Propagate handles this.
        
        ok, _ = propagate()
        if not ok:
            return False, None
        
        if not active:
            return True, None
            
        v = choose_var()
        if v == -1:
            return True, None
        
        trail_len = len(trail)
        
        # Try v = True
        values[v] = 1
        trail.append((v, 1))
        # Update state
        old_active = active
        old_sizes = clause_size[:]
        
        sat_mask = pos[v] & active
        active = active & ~sat_mask
        unsat_mask = neg[v] & active
        contradiction = False
        mm = unsat_mask
        while mm:
            bit = mm & -mm
            ci = bit.bit_length() - 1
            clause_size[ci] -= 1
            if clause_size[ci] == 0:
                contradiction = True
            mm ^= bit
        
        if not contradiction:
            ok, _ = backtrack(d + 1)
            if ok:
                return True, None
        
        # Undo all changes since trail_len
        # Restore values for vars in trail[trail_len:]
        for i in range(len(trail) - 1, trail_len - 1, -1):
            var, val = trail[i]
            values[var] = 0
        trail = trail[:trail_len]
        active = old_active
        clause_size = old_sizes
        
        # Try v = False
        values[v] = -1
        trail.append((v, -1))
        old_active = active
        old_sizes = clause_size[:]
        
        sat_mask = neg[v] & active
        active = active & ~sat_mask
        unsat_mask = pos[v] & active
        contradiction = False
        mm = unsat_mask
        while mm:
            bit = mm & -mm
            ci = bit.bit_length() - 1
            clause_size[ci] -= 1
            if clause_size[ci] == 0:
                contradiction = True
            mm ^= bit
            
        if not contradiction:
            ok, _ = backtrack(d + 1)
            if ok:
                return True, None
        
        # Undo all changes since trail_len
        for i in range(len(trail) - 1, trail_len - 1, -1):
            var, val = trail[i]
            values[var] = 0
        trail = trail[:trail_len]
        active = old_active
        clause_size = old_sizes
        
        return False, None

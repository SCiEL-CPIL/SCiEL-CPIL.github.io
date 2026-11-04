<<<<<<< SEARCH
    def fast_girth_check(m, voltages, g_target, time_limit=None):
        """
        Faster girth check for K_{3,3} voltage lift.
        Exploits the structure: the base is K_{3,3} which has girth 4.
        A cycle in the lift of length L corresponds to a closed walk of length L
        in K_{3,3} whose total voltage is 0 mod m.
        
        For girth >= g, we need no closed walk of length < g with voltage 0 mod m.
        The shortest closed walks in K_{3,3}:
        - Length 4: Ai-Bj-Ak-Bl-Ai (i!=k or j!=l... actually any 4-cycle)
          In K_{3,3}, 4-cycles are Ai-Bj-Ak-Bl-Ai where i!=k and j!=l.
          Voltage: v_{i,j} - v_{i,l} + v_{k,l} - v_{k,j} (depending on orientation)
          Actually let me think more carefully.
        
        Walk Ai -> Bj -> Ak -> Bl -> Ai:
          Ai to Bj: +v_{i,j}
          Bj to Ak: -v_{k,j}  (going from B to A, voltage is negated)
          Ak to Bl: +v_{k,l}
          Bl to Ai: -v_{i,l}
          Total: v_{i,j} - v_{k,j} + v_{k,l} - v_{i,l} = 0 mod m
        
        - Length 6: various 6-cycles in K_{3,3}
        
        This approach is complex. Let's just use direct BFS on the lift for small m,
        and for larger m, use the voltage sum approach.
        """
        # For m <= 100, direct BFS is feasible
        if m <= 120:
            return girth_of_lift(m, voltages, g_target, time_limit)
        
        # For larger m, check voltage sums for short closed walks
        # 4-cycles: for all i!=k, j!=l: v_{i,j} - v_{k,j} + v_{k,l} - v_{i,l} != 0 mod m
        for i in range(3):
            for k in range(3):
                if i == k:
                    continue
                for j in range(3):
                    for l in range(3):
                        if j == l:
                            continue
                        total = (voltages[i*3+j] - voltages[k*3+j] + voltages[k*3+l] - voltages[i*3+l]) % m
                        if total == 0:
                            return 4
        
        # 6-cycles in K_{3,3}: Ai-Bj-Ak-Bl-Am-Bn-Ai with distinct conditions
        # This is getting complex. Let's just do BFS for m up to 200.
        if m <= 200:
            return girth_of_lift(m, voltages, g_target, time_limit)
        
        # For larger m, we'll need a smarter approach
        # For now, return -1 (unknown) to indicate we can't verify
        return -1
=======
    def fast_girth_check(m, voltages, g_target, time_limit=None):
        """
        Fast girth check for K_{3,3} voltage lift using voltage sum method.
        Checks all closed walks of length < g_target in K_{3,3} base.
        """
        n = 6 * m
        if time_limit and time.time() > time_limit:
            return -1
        
        # 4-cycles: Ai-Bj-Ak-Bl-Ai where i!=k, j!=l
        # Voltage: v_{i,j} - v_{k,j} + v_{k,l} - v_{i,l} mod m
        for i in range(3):
            for k in range(3):
                if i == k:
                    continue
                for j in range(3):
                    for l in range(3):
                        if j == l:
                            continue
                        total = (voltages[i*3+j] - voltages[k*3+j] + voltages[k*3+l] - voltages[i*3+l]) % m
                        if total == 0:
                            return 4
        
        # 6-cycles in K_{3,3}: Ai-Bj-Ak-Bl-Am-Bn-Ai
        # where i,k,m are distinct and j,l,n are distinct (or some can repeat)
        # Actually in K_{3,3}, a 6-cycle visits 3 A-vertices and 3 B-vertices
        # Each A connects to exactly 2 of the 3 B's in the cycle
        # Enumerate: choose 3 A's (only 1 way since we have 3), 3 B's (only 1 way)
        # Then the cycle is a permutation matching
        # A0-B?-A1-B?-A2-B?-A0
        # The B's between must be distinct
        import itertools
        for perm in itertools.permutations([0, 1, 2]):
            # A0-Bperm[0]-A1-Bperm[1]-A2-Bperm[2]-A0
            total = (voltages[0*3+perm[0]] - voltages[1*3+perm[0]] 
                     + voltages[1*3+perm[1]] - voltages[2*3+perm[1]]
                     + voltages[2*3+perm[2]] - voltages[0*3+perm[2]]) % m
            if total == 0:
                return 6
        
        # 8-cycles: more complex. For g_target > 8, check 8-cycles
        if g_target > 8:
            # 8-cycles in K_{3,3}: visit some A and B vertices, possibly repeating
            # This gets very complex. For m <= 200, fall back to BFS.
            if m <= 200:
                return girth_of_lift(m, voltages, g_target, time_limit)
            # For larger m, we can't easily enumerate 8-cycles
            # Return a conservative estimate: if no 4 or 6 cycles found, girth >= 7
            # But we need to be sure about 8, 10, etc.
            # Return -1 to indicate unknown
            return -1
        
        return 8  # No 4 or 6 cycles found, so girth >= 7, and since bipartite, >= 8
>>>>>>> REPLACE


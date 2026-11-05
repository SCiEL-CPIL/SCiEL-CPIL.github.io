    def get_checkers_minus(s, checkers, u):
        sums3, F, sums_with, contrib = checkers
        new_s = s - {u}
        new_sums3 = sums3 - sums_with[u]
        
        # Recompute F for new_s
        new_F = {}
        lst = sorted(new_s)
        for a in lst:
            forbidden = set()
            others = [x for x in lst if x != a]
            m = len(others)
            for i in range(m):
                for j in range(i + 1, m):
                    residue = -(others[i] + others[j]) % a
                    forbidden.add(residue)
            new_F[a] = forbidden
            
        # We don't strictly need sums_with and contrib for the can_add check of v,
        # unless we plan to do another swap immediately.
        # can_add only needs sums3 and F.
        return (new_sums3, new_F, {}, {})

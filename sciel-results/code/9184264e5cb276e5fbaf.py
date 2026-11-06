    def can_add(a, S):
        rem_counts = {}
        for x in S:
            if x > a:
                r = x % a
                rem_counts[r] = rem_counts.get(r, 0) + 1
        
        for r, count in rem_counts.items():
            if (2 * r) % a == 0:
                return False
        
        present_rems = list(rem_counts.keys())
        for r1 in present_rems:
            r2 = (-r1) % a
            if r1 == r2:
                continue
            if r2 in rem_counts:
                return False
                
        return True

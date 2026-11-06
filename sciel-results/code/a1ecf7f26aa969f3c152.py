import time


def construct():
        start_time = time.time()
        deadline = start_time + 85.0
        n = 256
        
        S = []
        # We maintain S as a list of elements in descending order.
        # When adding x, all elements in S are > x.
        
        for x in range(n, 0, -1):
            if time.time() > deadline:
                break
            
            # Check if x divides any sum b+c where b, c in S
            # Since S is built from larger numbers, b,c > x.
            # We need to check all pairs (b, c) in S.
            # Optimization: If len(S) is small, O(k^2) is fine.
            # Max len(S) is 256. 256^2 = 65k ops.
            # 256 iterations * 65k = 16M ops.
            # Python handles ~10-50M simple ops/sec.
            # 16M might take 1-2 seconds. This is well within 90s.
            
            valid = True
            k = len(S)
            for i in range(k):
                b = S[i]
                # c must be >= b. Since S is sorted desc, S[i] >= S[j] for j > i.
                # Wait, S is descending: S[0] > S[1] > ...
                # So b = S[i]. c can be S[j] for j >= i.
                # But we just need *any* pair.
                # Actually, the condition is a < b <= c.
                # Here a=x. b and c are from S.
                # Since all elements in S are > x, the condition a < b is always true for any b in S.
                # We just need b <= c.
                # So we check all unordered pairs {b, c} from S.
                
                for j in range(i, k):
                    c = S[j]
                    if (b + c) % x == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                S.append(x)
        
        return sorted(S)

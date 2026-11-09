import numpy as np

def construct():
    N = 256
    
    # Precompute 4-AP partners
    # partners[x][y] = list of (a, b) such that {x, y, a, b} is a 4-AP
    # We only need to store pairs {a,b} with a < b
    partners = [[[] for _ in range(N + 1)] for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            # Check all 6 relative positions of x and y in a 4-AP
            # AP: t, t+d, t+2d, t+3d
            # Positions: 0, 1, 2, 3
            
            # Case 1: x=pos 0, y=pos 1 => t=x, d=y-x
            # Others: t+2d = x+2(y-x) = 2y-x, t+3d = x+3(y-x) = 3y-2x
            d = y - x
            if d > 0:
                a = x + 2 * d
                b = x + 3 * d
                if a <= N and b <= N:
                    partners[x][y].append((a, b))
                    partners[y][x].append((a, b))
                    
            # Case 2: x=pos 0, y=pos 2 => t=x, 2d=y-x => d=(y-x)/2
            if (y - x) % 2 == 0:
                d = (y - x) // 2
                if d > 0:
                    a = x + d
                    b = x + 3 * d
                    if a <= N and b <= N:
                        partners[x][y].append((a, b))
                        partners[y][x].append((a, b))
                        
            # Case 3: x=pos 0, y=pos 3 => t=x, 3d=y-x => d=(y-x)/3
            if (y - x) % 3 == 0:
                d = (y - x) // 3
                if d > 0:
                    a = x + d
                    b = x + 2 * d
                    if a <= N and b <= N:
                        partners[x][y].append((a, b))
                        partners[y][x].append((a, b))
                        
            # Case 4: x=pos 1, y=pos 2 => t+ d = x, t+2d=y => d=y-x, t=x-d
            d = y - x
            if d > 0:
                a = x - d
                b = y + d
                if a >= 1 and b <= N:
                    partners[x][y].append((a, b))
                    partners[y][x].append((a, b))
                    
            # Case 5: x=pos 1, y=pos 3 => t+d=x, t+3d=y => 2d=y-x => d=(y-x)/2
            if (y - x) % 2 == 0:
                d = (y - x) // 2
                if d > 0:
                    a = x - d
                    b = y + d
                    if a >= 1 and b <= N:
                        partners[x][y].append((a, b))
                        partners[y][x].append((a, b))
                        
            # Case 6: x=pos 2, y=pos 3 => t+2d=x, t+3d=y => d=y-x, t=x-2d
            d = y - x
            if d > 0:
                a = x - 2 * d
                b = x - d
                if a >= 1 and b <= N:
                    partners[x][y].append((a, b))
                    partners[y][x].append((a, b))

    S = set()
    blocked = set()
    
    # Precompute 3-AP partners for gain calculation?
    # gain(x) = count of a in S such that there exists b in S making {x,a,b} a 3-AP.
    # For fixed x, a, the possible b's are determined.
    # b = x - 2*(a-x) ? No.
    # 3-AP: u, u+d, u+2d.
    # If x and a are in the AP:
    # 1. x=u, a=u+d => d=a-x, b=u+2d = x+2(a-x) = 2a-x
    # 2. x=u, a=u+2d => 2d=a-x, d=(a-x)/2, b=u+d = x+(a-x)/2
    # 3. x=u+d, a=u => d=x-a, b=u+2d = a+2(x-a) = 2x-a
    # 4. x=u+d, a=u+2d => d=a-x, b=u = x-d = 2x-a
    # 5. x=u+2d, a=u => 2d=x-a, d=(x-a)/2, b=u+d = a+(x-a)/2
    # 6. x=u+2d, a=u+d => d=x-a, b=u = x-d = 2x-a
    
    # So for each pair (x, a), there are up to 3 distinct candidates for b.
    # We just need to check if b is in S.
    
    def calc_gain(x, S):
        gain = 0
        for a in S:
            if a == x:
                continue
            # Generate possible b's
            # Case 1: x, a, b with x=1st, a=2nd => b = 2a - x
            b = 2 * a - x
            if 1 <= b <= N and b in S and b != a and b != x:
                gain += 1
            # Case 2: x, a, b with x=1st, a=3rd => b = (x+a)/2
            if (x + a) % 2 == 0:
                b = (x + a) // 2
                if 1 <= b <= N and b in S and b != a and b != x:
                    gain += 1
            # Case 3: x, a, b with x=2nd, a=1st => b = 2x - a
            b = 2 * x - a
            if 1 <= b <= N and b in S and b != a and b != x:
                gain += 1
            # Case 4: x, a, b with x=2nd, a=3rd => b = 2x - a (same as above? No)
            # If x=2nd, a=3rd, then x=u+d, a=u+2d. d=x-u, a=x+d => a=2x-u => u=2x-a.
            # The third term is u+2d? No, the AP is u, u+d, u+2d.
            # Elements are x, a.
            # If x is 2nd, a is 3rd: terms are b, x, a.
            # x = b+d, a=x+d => a=b+2d => d=(a-b)/2.
            # Also x=b+d => b=x-d.
            # From x=b+d and a=b+2d => a-x=d. So d=a-x.
            # b = x - d = x - (a-x) = 2x - a.
            # This is the same formula as Case 3?
            # Case 3: x=2nd, a=1st. Terms are a, x, b.
            # x=a+d, b=x+d => b=a+2d. d=x-a. b=a+2(x-a)=2x-a.
            # Yes, same b.
            # But we must ensure the AP is non-constant and distinct.
            # The gain counts distinct 3-APs.
            # A 3-AP is defined by its set of 3 elements.
            # So for a fixed pair {x, a}, there is at most one b that completes a 3-AP?
            # No.
            # Example: {1, 3, 5}. Pair {1, 5} -> b=3. Pair {1, 3} -> b=5 or b=-1.
            # So for a fixed pair {x, a}, there can be multiple b's?
            # No, an arithmetic progression is determined by 2 terms and their positions.
            # But the set {x, a, b} is just a set.
            # Does {1, 3, 5} count as one 3-AP? Yes.
            # If we have {1, 3, 5}, and we add 5.
            # Pairs in S: {1, 3}.
            # b for {1, 3} can be 5 (1,3,5) or -1 (invalid).
            # So gain is 1.
            # What if S={1, 5, 3}?
            # Pairs: {1, 5} -> b=3. {1, 3} -> b=5. {5, 3} -> b=1.
            # All point to the same set {1, 3, 5}.
            # But we are counting *new* 3-APs formed by adding x.
            # So we count pairs {a, b} in S such that {x, a, b} is a 3-AP.
            # If S={1, 3} and we add 5:
            # Pair {1, 3} completes to {1, 3, 5}. Gain 1.
            # Pair {1, 5} is not in S.
            # So we just iterate pairs? No, iterate a in S, find b in S.
            # To avoid double counting, we can just check if the set {x, a, b} is a 3-AP.
            # Since we iterate a, we might find the same AP twice?
            # If we add x, and S has {a, b}.
            # We check if {x, a, b} is a 3-AP.
            # This is a property of the set.
            # So for each a in S, we check if there exists b in S such that {x, a, b} is a 3-AP.
            # If yes, we add 1 to gain.
            # But if {x, a, b} and {x, a, c} are two different APs?
            # Yes, possible.
            # Example: S={2, 4, 6}. Add x=1.
            # {1, 2, ?} -> 3 (not in S).
            # {1, 4, ?} -> ?
            # Let's just use the set check.
            # For each a in S:
            #   Calculate potential b's.
            #   If b in S, then {x, a, b} is a 3-AP.
            #   We need to ensure we don't count the same 3-AP multiple times.
            #   A 3-AP is uniquely identified by its min, max, and mid?
            #   Or just the set.
            #   If we find b for a given a, the set is {x, a, b}.
            #   If we find b' for a given a', the set is {x, a', b'}.
            #   If {x, a, b} == {x, a', b'}, it's the same AP.
            #   This happens if {a, b} == {a', b'}.
            #   So if we iterate a, we might count the same AP twice if we find it via a and via b.
            #   To fix: only count if a < b? Or use a set of counted APs.
            #   Given N=256, a set of counted APs is cheap.
            
        # Revised calc_gain
        counted_aps = set()
        for a in S:
            # Generate all b such that {x, a, b} is a 3-AP
            # We can just check the 3 possible b's derived from x, a
            b_candidates = set()
            # x=1, a=2 => b=2a-x
            b1 = 2 * a - x
            if 1 <= b1 <= N:
                b_candidates.add(b1)
            # x=1, a=3 => b=(x+a)/2
            if (x + a) % 2 == 0:
                b2 = (x + a) // 2
                if 1 <= b2 <= N:
                    b_candidates.add(b2)
            # x=2, a=1 => b=2x-a
            b3 = 2 * x - a
            if 1 <= b3 <= N:
                b_candidates.add(b3)
            
            for b in b_candidates:
                if b in S and b != a and b != x:
                    # Check if {x, a, b} is a valid non-constant 3-AP
                    # It is, by construction of b.
                    # Create a canonical key for the AP
                    ap_set = frozenset((x, a, b))
                    if ap_set not in counted_aps:
                        counted_aps.add(ap_set)
        return len(counted_aps)

    while True:
        best_x = None
        best_gain = -1
        
        # Iterate all candidates
        for x in range(1, N + 1):
            if x in S or x in blocked:
                continue
            
            gain = calc_gain(x, S)
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
            elif gain == best_gain:
                if best_x is None or x < best_x:
                    best_x = x
        
        if best_x is None:
            break
        
        S.add(best_x)
        
        # Update blocked
        # For each y not in S and not best_x
        # Check if y is now blocked due to best_x
        # y is blocked if there exists a, b in S_old such that {best_x, y, a, b} is a 4-AP
        S_old = S - {best_x}
        
        for y in range(1, N + 1):
            if y in S or y == best_x:
                continue
            if y in blocked:
                continue
            
            # Check partners[best_x][y]
            for a, b in partners[best_x][y]:
                if a in S_old and b in S_old:
                    blocked.add(y)
                    break

    return sorted(S)

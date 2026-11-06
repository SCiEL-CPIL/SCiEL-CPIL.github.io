import math

def construct():
    # 255 = 3 * 5 * 17. Every element must share a factor with 255.
    # The multiples of 3 in [1,255] form a clique (all share factor 3): 85 elements.
    # Any non-multiple-of-3 must be divisible by 5 AND 17 (=85) to connect to all
    # multiples of 3; the only such number <=255 is 85. So max clique = 86.
    result = [i for i in range(3, 256, 3)]
    result.append(85)
    result.append(255)  # 255 is already a multiple of 3; ensure present, dedup below

    # Deduplicate while preserving a valid set
    seen = set()
    dedup = []
    for x in result:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    result = dedup

    # Safety verification: every pair must have gcd > 1 and 255 must be present.
    ok = (255 in seen)
    n = len(result)
    if ok:
        for i in range(n):
            ai = result[i]
            for j in range(i + 1, n):
                if math.gcd(ai, result[j]) == 1:
                    ok = False
                    break
            if not ok:
                break

    if not ok:
        # Fallback: return all multiples of 3 (guaranteed valid clique, size 85)
        result = [i for i in range(3, 256, 3)]
        # ensure 255 present (it is, since 255 % 3 == 0)
    return result

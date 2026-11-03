def beatty(N, alpha):
    frac = np.mod(np.arange(N) * alpha, 1.0)
    b = (frac < 0.5).astype(float)
    # adjust count to N//2
    target = N // 2
    cur = b.sum()
    if cur > target:
        # flip some 1s to 0s: those with frac closest to 0.5 from above? Actually flip the 1s with largest frac (closest to 0.5 boundary on the high side)
        idx1 = np.where(b==1)[0]
        # flip the ones with frac closest to 0.5 (i.e., largest frac among 1s, since 1s have frac<0.5, largest is closest to 0.5)
        order = idx1[np.argsort(-frac[idx1])]
        for i in order[:int(cur-target)]:
            b[i] = 0
    elif cur < target:
        idx0 = np.where(b==0)[0]
        order = idx0[np.argsort(frac[idx0])]  # smallest frac, closest to 0 boundary → closest to 0.5 from below? no
        # 0s have frac>=0.5; closest to 0.5 are smallest frac among 0s
        for i in order[:int(target-cur)]:
            b[i] = 1
    return b

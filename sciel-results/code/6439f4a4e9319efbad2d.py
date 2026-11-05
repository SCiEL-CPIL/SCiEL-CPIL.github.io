random.seed(12345)
code_set = set(code.tolist())
stall = 0
KICK_EVERY = 400
while time.time() < deadline - 10 and len(code) > 0:
    i = random.randrange(len(code))
    c = int(code[i])
    nb = int(balls[c][1 + random.randrange(16)])
    if nb in code_set:
        continue
    bc = balls[c]; bcn = balls[nb]
    bcn_set = set(bcn.tolist())
    valid = True
    for w in bc.tolist():
        if w not in bcn_set and cover_count[w] < 2:
            valid = False; break
    if not valid:
        continue
    # apply
    cover_count[bc] -= 1
    cover_count[bcn] += 1
    code[i] = nb
    code_set.discard(c); code_set.add(nb)
    old_len = len(code)
    code = removal_pass(code, cover_count)
    if len(code) < old_len:
        stall = 0
    else:
        # revert
        cover_count[bcn] -= 1
        cover_count[bc] += 1
        code[i] = c
        code_set.discard(nb); code_set.add(c)
        stall += 1
        if stall >= KICK_EVERY:
            # kick: do a few random valid swaps to perturb
            for _ in range(3):
                if len(code) == 0: break
                j = random.randrange(len(code))
                cj = int(code[j])
                nbj = int(balls[cj][1 + random.randrange(16)])
                if nbj in code_set: continue
                bcj = balls[cj]; bcnj = balls[nbj]
                bcnj_set = set(bcnj.tolist())
                ok = True
                for w in bcj.tolist():
                    if w not in bcnj_set and cover_count[w] < 2:
                        ok = False; break
                if not ok: continue
                cover_count[bcj] -= 1
                cover_count[bcnj] += 1
                code[j] = nbj
                code_set.discard(cj); code_set.add(nbj)
            stall = 0

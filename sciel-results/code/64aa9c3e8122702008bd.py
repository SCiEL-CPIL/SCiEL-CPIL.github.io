def aggressive_remove_redundant(net):
    if not net:
        return net
    prev_len = len(net) + 1
    while len(net) < prev_len:
        prev_len = len(net)
        net, _ = remove_redundant(net)
    return net

def greedy_polish(net):
    if not net:
        return net
    changed = True
    while changed:
        changed = False
        for i in range(len(net) - 1, -1, -1):
            candidate = net[:i] + net[i+1:]
            if fast_verify(candidate):
                net = candidate
                changed = True
                break
    return net

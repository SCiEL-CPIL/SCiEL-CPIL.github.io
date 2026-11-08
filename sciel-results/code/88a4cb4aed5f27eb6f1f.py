def block_cost(si,sj,sk):
    if si==2 and sj==2 and sk==2: return 7
    if si==3 and sj==3 and sk==3: return 23
    if si==1 or sj==1 or sk==1: return si*sj*sk
    # split largest dim >=3
    best = si*sj*sk
    if si>=3: best=min(best, block_cost(2,sj,sk)+block_cost(si-2,sj,sk))
    if sj>=3: best=min(best, block_cost(si,2,sk)+block_cost(si,sj-2,sk))
    if sk>=3: best=min(best, block_cost(si,sj,2)+block_cost(si,sj,sk-2))
    return best

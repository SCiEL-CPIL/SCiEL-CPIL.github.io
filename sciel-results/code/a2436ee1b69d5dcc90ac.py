affected=[]
delta=0
for idx,slot in _POS_APS[p]:
    ...compute nv, ov=viol[idx]
    if nv!=ov:
        delta+=nv-ov
        affected.append((idx,nv))
new_total=total+delta
accept = ...
if accept:
    cols[p]=new
    for idx,nv in affected:
        viol[idx]=nv
    total=new_total
    ...

# For A=[[a,b],[c,d]], B=[[x,y,z],[u,v,w]], C=A*B
# C00=ax+by, C01=ay+bz, C02=az+bw
# C10=cx+dy, C11=cy+dz, C12=cz+dw

# 10 rank-1 terms:
# T1: (a+c)(x+u) -> C00+=1, C10+=1
# T2: c(x+u)     -> C10+=1, C00-=1  [so net: C00 from T1-T2, C10 from T1+T2]
# ... this is getting complicated. Let me just use a known correct one.

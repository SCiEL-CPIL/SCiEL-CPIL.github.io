import inspect
import importlib

try:
    mod = importlib.import_module("autoresearch_suite.science_tcs_expansion")
    print("MODULE OK")
    for name in dir(mod):
        if not name.startswith("__"):
            print("  ", name)
    for fn in ["pebble_baseline"]:
        if hasattr(mod, fn):
            f = getattr(mod, fn)
            print("\n---", fn, "---")
            try:
                print(inspect.getsource(f))
            except Exception as e:
                print("no source:", e)
except Exception as e:
    print("IMPORT FAILED:", repr(e))
